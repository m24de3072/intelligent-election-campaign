#!/usr/bin/env python3
"""
add_triplet_urban_flag.py

Takes:
 - ./election_outputs/electoral_with_census_attributes.csv
 - /path/to/Census.csv

Produces:
 - ./election_outputs/electoral_with_census_attributes_with_urban_flag.csv

For each electoral row it looks up the census triplet (Total / Rural / Urban)
by matching the c_<Name> value to Census 'Name' and using the Census level column
(which contains 'Total' / 'Rural' / 'Urban'). It then compares urban vs rural population
and sets c_is_urban_triplet = 1/0/-1.
"""
import re
from pathlib import Path
import pandas as pd
import numpy as np

# ---------- CONFIG - adjust paths if needed ----------
ELECTORAL_PATH = Path("./election_outputs/electoral_with_census_attributes.csv")
CENSUS_PATH = Path("./Census.csv")  # replace if needed
OUT_DIR = Path("./election_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "electoral_with_census_attributes_with_urban_flag.csv"

# ---------- helpers ----------
def normalize_name(s):
    if pd.isna(s): return ""
    s = str(s).strip().lower()
    s = s.encode('ascii', 'ignore').decode('ascii')
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    for k,v in cols.items():
        for cand in candidates:
            if cand.lower() in k:
                return v
    return None

def detect_triplet_label_col(df):
    """
    Find the column that contains Total / Rural / Urban labels (returns column name or None).
    """
    target = {'total','rural','urban'}
    for c in df.columns:
        try:
            vals = df[c].astype(str).str.lower().unique()
        except Exception:
            continue
        found = set()
        for v in vals:
            for t in target:
                if t in v:
                    found.add(t)
            if len(found) == 3:
                return c
        if len(found) >= 2:  # accept >=2 (Total+Urban / Total+Rural)
            return c
    return None

def detect_population_col(df):
    """
    Heuristic to choose the column representing total population.
    """
    candidates = ['total population','total_pop','total_population','population','persons','tot_p','population_total','pop_total']
    for c in df.columns:
        low = c.lower()
        for cand in candidates:
            if cand in low:
                return c
    # fallback: numeric column with largest sum
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        # try coercing numeric-like columns
        maybe = []
        for c in df.columns:
            try:
                s = pd.to_numeric(df[c].astype(str).str.replace(',',''), errors='coerce')
                if s.notna().sum() > 0:
                    maybe.append((s.sum(), c))
            except Exception:
                continue
        if maybe:
            maybe.sort(reverse=True)
            return maybe[0][1]
        return None
    sums = {c: pd.to_numeric(df[c], errors='coerce').fillna(0).sum() for c in numeric_cols}
    best = max(sums.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else None

def to_num(x):
    try:
        if pd.isna(x): return np.nan
        s = str(x).replace(',', '').strip()
        return float(s) if s != '' else np.nan
    except Exception:
        return np.nan

# ---------- load files ----------
print("Loading files...")
if not ELECTORAL_PATH.exists():
    raise FileNotFoundError(f"Electoral file not found at {ELECTORAL_PATH}")
if not CENSUS_PATH.exists():
    raise FileNotFoundError(f"Census file not found at {CENSUS_PATH}")

elect = pd.read_csv(ELECTORAL_PATH, low_memory=False)
census = pd.read_csv(CENSUS_PATH, low_memory=False)
print("Rows loaded -> electoral:", len(elect), "census:", len(census))

# ---------- detect census name & triplet label & population columns ----------
# detect census name column in raw census
census_name_col = find_col(census, ['name','place','location','constituency','area','sub-district','sub district'])
if census_name_col is None:
    census_name_col = census.columns[0]
print("Detected census name column:", census_name_col)

# Force triplet column explicitly
triplet_col = "Total/\nRural/\nUrban"   # exact column name in Census.csv

if triplet_col not in census.columns:
    raise KeyError(f"Triplet column '{triplet_col}' not found in Census dataset. "
                   f"Available columns: {list(census.columns)}")
print("Using triplet label column:", triplet_col)

pop_col = detect_population_col(census)
print("Detected population column:", pop_col)

# ---------- find corresponding c_<Name> column in electoral file ----------
# candidate c_ column name may be 'c_<census_name_col>' or 'c_Name' etc.
possible_cname_cols = []
candidates_to_try = [
    f"c_{census_name_col}",
    f"c_{census_name_col}".replace(' ', '_'),
    f"c_{census_name_col}".lower(),
    "c_name",
    "c_Name",
    "c_CENSUS_NAME",
]
for col in candidates_to_try:
    if col in elect.columns:
        possible_cname_cols.append(col)
# if not found, search for any column in elect that startswith 'c_' and contains 'name' token
if not possible_cname_cols:
    for col in elect.columns:
        if col.startswith('c_') and 'name' in col.lower():
            possible_cname_cols.append(col)
# final fallback: try to infer from column names matching census_name_col substring
if not possible_cname_cols:
    for col in elect.columns:
        if census_name_col.lower() in col.lower():
            possible_cname_cols.append(col)

if not possible_cname_cols:
    raise RuntimeError("Could not find c_<Name> column in electoral file. Columns available: " + ", ".join(elect.columns[:50]))

c_name_col = possible_cname_cols[0]
print("Using electoral column for matched census name:", c_name_col)

# ---------- build lookup: census rows grouped by normalized name ----------
census['name_norm'] = census[census_name_col].astype(str).map(normalize_name)
rows_by_name = {}
for idx, r in census.iterrows():
    k = r['name_norm']
    rows_by_name.setdefault(k, []).append(idx)

# ---------- Prepare output columns in electoral DF ----------
elect['c_total_pop_triplet'] = np.nan
elect['c_urban_pop_triplet'] = np.nan
elect['c_rural_pop_triplet'] = np.nan
elect['c_urban_pct_triplet'] = np.nan
elect['c_is_urban_triplet'] = -1   # 1 urban, 0 rural, -1 unknown

# ---------- iterate electoral rows and fetch triplet from census ----------
for i, row in elect.iterrows():
    c_name_val = row.get(c_name_col, "")
    if pd.isna(c_name_val) or str(c_name_val).strip() == "":
        # no c_name value
        elect.at[i, 'c_is_urban_triplet'] = -1
        continue
    name_norm = normalize_name(c_name_val)
    census_idxs = rows_by_name.get(name_norm, [])
    if not census_idxs:
        # no match found
        elect.at[i, 'c_is_urban_triplet'] = -1
        continue

    # among census_idxs find rows labelled Total / Urban / Rural via triplet_col
    total_idx = None
    urban_idx = None
    rural_idx = None

    for ci in census_idxs:
        if triplet_col:
            label = str(census.at[ci, triplet_col]).lower()
            if 'total' in label:
                total_idx = ci
            elif 'urban' in label:
                urban_idx = ci
            elif 'rural' in label:
                rural_idx = ci
        else:
            # if no triplet column detected, try to detect by searching textual values in the whole row
            row_text = " ".join([str(census.at[ci, c]) for c in census.columns if isinstance(census.at[ci,c], str)])
            row_text = row_text.lower()
            if 'total' in row_text:
                total_idx = ci
            elif 'urban' in row_text:
                urban_idx = ci
            elif 'rural' in row_text:
                rural_idx = ci

    # if missing any of urban/rural, attempt to find rows whose 'Name' contains 'urban'/'rural' suffix
    if (urban_idx is None or rural_idx is None) and len(census_idxs) > 1:
        for ci in census_idxs:
            nm = str(census.at[ci, census_name_col]).lower()
            # sometimes census name may include "(Urban)" or "- Urban"
            if 'urban' in nm and urban_idx is None:
                urban_idx = ci
            if 'rural' in nm and rural_idx is None:
                rural_idx = ci

    # get population values (try pop_col)
    total_pop = to_num(census.at[total_idx, pop_col]) if total_idx is not None and pop_col in census.columns else np.nan
    urban_pop = to_num(census.at[urban_idx, pop_col]) if urban_idx is not None and pop_col in census.columns else np.nan
    rural_pop = to_num(census.at[rural_idx, pop_col]) if rural_idx is not None and pop_col in census.columns else np.nan

    # If pop_col not found or NaNs, try to find the largest numeric cell in each row as fallback
    def fallback_row_pop(ci):
        if ci is None:
            return np.nan
        # choose largest numeric value in that row
        nums = []
        for c in census.columns:
            try:
                v = to_num(census.at[ci, c])
                if not pd.isna(v):
                    nums.append(v)
            except Exception:
                continue
        return max(nums) if nums else np.nan

    if pd.isna(urban_pop):
        urban_pop = fallback_row_pop(urban_idx)
    if pd.isna(rural_pop):
        rural_pop = fallback_row_pop(rural_idx)
    if pd.isna(total_pop):
        total_pop = fallback_row_pop(total_idx)

    # store in electoral df
    elect.at[i, 'c_total_pop_triplet'] = total_pop
    elect.at[i, 'c_urban_pop_triplet'] = urban_pop
    elect.at[i, 'c_rural_pop_triplet'] = rural_pop

    # determine urban pct and flag
    if (not pd.isna(urban_pop)) and (not pd.isna(rural_pop)) and (urban_pop + rural_pop > 0):
        u_pct = urban_pop / (urban_pop + rural_pop)
        elect.at[i, 'c_urban_pct_triplet'] = u_pct
        elect.at[i, 'c_is_urban_triplet'] = 1 if urban_pop >= rural_pop else 0
    else:
        elect.at[i, 'c_urban_pct_triplet'] = np.nan
        elect.at[i, 'c_is_urban_triplet'] = -1

# ---------- save updated electoral file ----------
elect.to_csv(OUT_PATH, index=False)
print("Saved updated electoral file with triplet urban flag to:", OUT_PATH)
print("Summary:")
print(" total electoral rows:", len(elect))
print(" rows with known urban/rural flag:", (elect['c_is_urban_triplet'] != -1).sum())
print(" rows marked urban (1):", (elect['c_is_urban_triplet'] == 1).sum())
print(" rows marked rural (0):", (elect['c_is_urban_triplet'] == 0).sum())
print(" rows unknown (-1):", (elect['c_is_urban_triplet'] == -1).sum())
