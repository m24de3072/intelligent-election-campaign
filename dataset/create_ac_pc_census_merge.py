#!/usr/bin/env python3
"""
map_electoral_to_census_reverse.py

For each electoral (TCPD) row, find the best matching Census row (District or Sub-district)
and attach the matched census columns (prefixed with c_) to the electoral row.

Inputs (adjust if needed):
 - /mnt/data/TCPD_GA_All_States_2025-10-7.csv
 - /mnt/data/Census.csv

Output:
 - ./election_outputs/electoral_with_census_attributes.csv
"""
import re
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
TCPD_PATH = Path("/Users/priyeashbala/Personal_Projects/Smart_Election_Campaign/TCPD_GA_All_States_2025-10-7.csv")
CENSUS_PATH = Path("/Users/priyeashbala/Personal_Projects/Smart_Election_Campaign/Census.csv")
OUT_DIR = Path("./election_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

USE_FUZZY = True       # set False to disable fuzzy matching
FUZZY_THRESHOLD = 85   # 0-100 (higher -> stricter)

# optional fuzzy library
try:
    from rapidfuzz import process, fuzz
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False
    if USE_FUZZY:
        print("rapidfuzz not installed; fuzzy matching will be disabled. Install via `pip install rapidfuzz` to enable fuzzy matches.")
        USE_FUZZY = False

# ---------- HELPERS ----------
def normalize_name(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.encode('ascii', 'ignore').decode('ascii')
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    for k, v in cols.items():
        for cand in candidates:
            if cand.lower() in k:
                return v
    return None

# ---------- LOAD ----------
print("Loading files...")
tcpd = pd.read_csv(TCPD_PATH, low_memory=False)
census = pd.read_csv(CENSUS_PATH, low_memory=False)
print("Loaded rows -> TCPD:", len(tcpd), "Census:", len(census))

# ---------- DETECT COLUMNS ----------
# electoral columns (the ones you showed: PC_Name, PC_No, Constituency_Name)
pc_name_col = find_col(tcpd, ['pc_name','pc_name_orig','pc_name_eng','pc'])
pc_no_col   = find_col(tcpd, ['pc_no','pc_number','pc'])
ac_col      = find_col(tcpd, ['constituency','constituency_name','Constituency_Name','ac_name','assembly_constituency'])
state_col_tcpd = find_col(tcpd, ['state','state_name','st_name'])

# census columns
census_name_col = find_col(census, ['name','place','location']) or census.columns[0]
# the "India/ State/ Union Territory/ District/ Sub-district" column
census_level_col = find_col(census, ['india/ state/ union territory/ district/ sub-district','level','area_type','admin_level','type','india'])
state_col_census = find_col(census, ['state','state_name'])

print("Detected columns:")
print(" TCPD => pc_name:", pc_name_col, "pc_no:", pc_no_col, "ac:", ac_col, "state:", state_col_tcpd)
print(" Census => name:", census_name_col, "level:", census_level_col, "state:", state_col_census)

# ---------- NORMALIZE NAMES ----------
# Prepare normalized fields in TCPD
tcpd['pc_name_orig'] = tcpd[pc_name_col].astype(str) if pc_name_col else ""
tcpd['pc_name_norm'] = tcpd['pc_name_orig'].map(normalize_name)
tcpd['pc_no_orig'] = tcpd[pc_no_col] if pc_no_col else None
if ac_col:
    tcpd['ac_orig'] = tcpd[ac_col].astype(str)
    tcpd['ac_norm'] = tcpd['ac_orig'].map(normalize_name)
else:
    tcpd['ac_orig'] = ""
    tcpd['ac_norm'] = ""
tcpd['state_norm'] = tcpd[state_col_tcpd].astype(str).map(normalize_name) if state_col_tcpd else ""

# Prepare normalized fields in Census
census['census_name_orig'] = census[census_name_col].astype(str)
census['census_name_norm'] = census['census_name_orig'].map(normalize_name)
census['state_norm'] = census[state_col_census].astype(str).map(normalize_name) if state_col_census else ""
if census_level_col:
    census['level_raw'] = census[census_level_col].astype(str)
    census['level_norm'] = census['level_raw'].astype(str).str.strip().str.lower()
else:
    census['level_raw'] = ""
    census['level_norm'] = ""

# classify census rows as District / Sub-district using level_norm heuristics
def is_district(s):
    if not s: return False
    s = str(s).lower()
    return 'district' in s and 'sub' not in s

def is_subdistrict(s):
    if not s: return False
    s = str(s).lower()
    return ('sub-district' in s) or ('sub district' in s) or ('subdistrict' in s) or any(k in s for k in ['tehsil','taluk','block','mandal'])

census['is_district'] = census['level_norm'].apply(is_district)
census['is_subdistrict'] = census['level_norm'].apply(is_subdistrict)

# fallback: if no rows detected as district/subdistrict, treat all rows as possible matches
if census['is_district'].sum() + census['is_subdistrict'].sum() == 0:
    print("Warning: no district/sub-district rows detected by level column heuristics. Using all census rows as candidate matches.")
    census['is_district'] = True
    census['is_subdistrict'] = True

# build lookup maps: normalized_name -> list of indices
district_rows = census[census['is_district']].copy()
subdistrict_rows = census[census['is_subdistrict']].copy()
district_map = {}
for idx, row in district_rows.iterrows():
    k = row['census_name_norm']
    if not k: continue
    district_map.setdefault(k, []).append(idx)
subdistrict_map = {}
for idx, row in subdistrict_rows.iterrows():
    k = row['census_name_norm']
    if not k: continue
    subdistrict_map.setdefault(k, []).append(idx)

# prepare fuzzy choices if needed
if USE_FUZZY and RAPIDFUZZ_AVAILABLE:
    district_choices = list(district_map.keys())
    subdistrict_choices = list(subdistrict_map.keys())

# ---------- MAPPING: electoral -> census (per electoral row) ----------
print("Mapping electoral rows -> census rows (prefer sub-district matches, then district)...")
mapped_idx = []
mapped_level = []
mapped_name = []
mapped_score = []

if USE_FUZZY and RAPIDFUZZ_AVAILABLE:
    from rapidfuzz import process, fuzz
else:
    process = None
    fuzz = None


for i, row in tcpd.iterrows():
    ac_norm = row.get('ac_norm', '')              # primary key to match
    state_norm = row.get('state_norm', '')        # to prefer same-state matches
    pc_norm = row.get('pc_name_norm', '')         # fallback matching using PC
    chosen_idx = None
    chosen_level = None
    chosen_name = None
    chosen_score = 0

    # 1) exact match on sub-district
    if ac_norm and ac_norm in subdistrict_map:
        cand_idxs = subdistrict_map[ac_norm]
        # prefer same state if possible
        chosen = None
        if state_norm:
            for ci in cand_idxs:
                if census.at[ci,'state_norm'] == state_norm:
                    chosen = ci
                    break
        if chosen is None:
            chosen = cand_idxs[0]
        chosen_idx = chosen
        chosen_level = 'Sub-district'
        chosen_name = census.at[chosen,'census_name_orig']
        chosen_score = 100
    # 2) exact match on district
    elif ac_norm and ac_norm in district_map:
        cand_idxs = district_map[ac_norm]
        chosen = None
        if state_norm:
            for ci in cand_idxs:
                if census.at[ci,'state_norm'] == state_norm:
                    chosen = ci
                    break
        if chosen is None:
            chosen = cand_idxs[0]
        chosen_idx = chosen
        chosen_level = 'District'
        chosen_name = census.at[chosen,'census_name_orig']
        chosen_score = 100
    else:
        # 3) fuzzy match subdistrict (if enabled)
        if USE_FUZZY and RAPIDFUZZ_AVAILABLE and ac_norm:
            # prefer subdistricts first
            if subdistrict_choices:
                res = process.extractOne(ac_norm, subdistrict_choices, scorer=fuzz.ratio)
                if res and res[1] >= FUZZY_THRESHOLD:
                    matched_norm = res[0]
                    cand_idxs = subdistrict_map[matched_norm]
                    chosen = None
                    if state_norm:
                        for ci in cand_idxs:
                            if census.at[ci,'state_norm'] == state_norm:
                                chosen = ci
                                break
                    if chosen is None:
                        chosen = cand_idxs[0]
                    chosen_idx = chosen
                    chosen_level = 'Sub-district (fuzzy)'
                    chosen_name = census.at[chosen,'census_name_orig']
                    chosen_score = res[1]
            # 4) fuzzy match district
            if chosen_idx is None and district_choices:
                res2 = process.extractOne(ac_norm, district_choices, scorer=fuzz.ratio)
                if res2 and res2[1] >= FUZZY_THRESHOLD:
                    matched_norm = res2[0]
                    cand_idxs = district_map[matched_norm]
                    chosen = None
                    if state_norm:
                        for ci in cand_idxs:
                            if census.at[ci,'state_norm'] == state_norm:
                                chosen = ci
                                break
                    if chosen is None:
                        chosen = cand_idxs[0]
                    chosen_idx = chosen
                    chosen_level = 'District (fuzzy)'
                    chosen_name = census.at[chosen,'census_name_orig']
                    chosen_score = res2[1]

    # 5) fallback: try matching by PC name (pc_norm) to either subdistrict or district
    if chosen_idx is None and pc_norm:
        if pc_norm in subdistrict_map:
            chosen_idx = subdistrict_map[pc_norm][0]
            chosen_level = 'Sub-district (by PC)'
            chosen_name = census.at[chosen_idx,'census_name_orig']
            chosen_score = 90
        elif pc_norm in district_map:
            chosen_idx = district_map[pc_norm][0]
            chosen_level = 'District (by PC)'
            chosen_name = census.at[chosen_idx,'census_name_orig']
            chosen_score = 90

    # 6) if still no match, leave None
    if chosen_idx is None:
        mapped_idx.append(None)
        mapped_level.append(None)
        mapped_name.append(None)
        mapped_score.append(0)
    else:
        mapped_idx.append(int(chosen_idx))
        mapped_level.append(chosen_level)
        mapped_name.append(chosen_name)
        mapped_score.append(chosen_score)

# attach mapping to tcpd
tcpd['census_idx'] = mapped_idx
tcpd['census_match_level'] = mapped_level
tcpd['census_match_name'] = mapped_name
tcpd['census_match_score'] = mapped_score

# Create c_ prefixed census columns in TCPD and fill for matched rows
for col in census.columns:
    tcpd['c_' + str(col)] = np.nan

for i, r in tcpd.iterrows():
    cidx = r['census_idx']
    if pd.isna(cidx) or cidx is None:
        continue
    cidx = int(cidx)
    for col in census.columns:
        tcpd.at[i, 'c_' + str(col)] = census.at[cidx, col]

# Save result
out_csv = OUT_DIR / "electoral_with_census_attributes.csv"
tcpd.to_csv(out_csv, index=False)
print("Saved electoral rows enriched with census attributes to:", out_csv)

# Save unmatched census rows sample for inspection
matched_census_idxs = set([int(x) for x in tcpd['census_idx'].dropna().unique() if x is not None])
all_census_idxs = set(census.index.tolist())
unmatched_idxs = sorted(list(all_census_idxs - matched_census_idxs))
unmatched_sample = census.loc[unmatched_idxs].head(500)
unmatched_sample.to_csv(OUT_DIR / "census_unmatched_sample.csv", index=False)
print("Saved sample of census rows not matched to electoral data to:", OUT_DIR / "census_unmatched_sample.csv")

# Summary
total = len(tcpd)
matched = tcpd['census_idx'].notna().sum()
print(f"Mapping complete: {matched}/{total} electoral rows matched ({matched/total:.2%}).")
print("Match level counts:")
print(tcpd['census_match_level'].value_counts(dropna=False))
