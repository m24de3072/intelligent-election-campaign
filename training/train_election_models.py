#!/usr/bin/env python3
"""
train_election_models_pc_aggregation_fixed.py

Fixed end-to-end training pipeline (PC aggregation, features, multiple models, ensemble).
Input:
 - /mnt/data/electoral_with_census_attributes_with_urban_flag.csv

Outputs (training/election_outputs):
 - recommended_candidates_pc.csv
 - pc_swing_ranking.csv
 - df_latest_predictions.csv
 - model_metrics.json
 - models/*.joblib
 - preprocessing.joblib

Notes:
 - This script is defensive: it aligns features used at prediction time with training-time features,
   handles unseen labels safely, and persists preprocessing artifacts (scaler, encoders, feature lists).
 - If you want TabNet, install pytorch-tabnet and set TABNET_AVAILABLE True.
"""
import re
import warnings
warnings.filterwarnings("ignore")
import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Optional libs
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except Exception:
    TabNetClassifier = None
    TABNET_AVAILABLE = False

# ---------- Paths ----------
DATA_PATH = Path("./dataset/election_outputs/electoral_with_census_attributes_with_urban_flag.csv")
OUT_DIR = Path("./election_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = OUT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---------- Helpers ----------
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

def safe_num(s):
    return pd.to_numeric(s.astype(str).str.replace(',','').str.replace(' ',''), errors='coerce')

def eval_model_probs(model, X, y):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:,1]
    else:
        probs = model.predict(X)
    preds = (probs >= 0.5).astype(int)
    res = {
        'accuracy': accuracy_score(y, preds),
        'precision': precision_score(y, preds, zero_division=0),
        'recall': recall_score(y, preds, zero_division=0),
        'f1': f1_score(y, preds, zero_division=0),
        'roc_auc': roc_auc_score(y, probs) if len(np.unique(y))>1 else None
    }
    return res, probs

def safe_label_transform_for_train(series):
    """Return cleaned strings for encoder fit/transform: fillna and str-cast."""
    return series.astype(str).fillna("NA")

def safe_label_transform_for_apply(le, series):
    """
    Transform with LabelEncoder 'le' safely: unseen labels mapped to fallback class.
    Returns numpy array same length as series.
    """
    vals = series.astype(str).fillna("NA").copy()
    if hasattr(le, 'classes_'):
        known = set([str(x) for x in le.classes_])
        fallback = 'NA' if 'NA' in known else (le.classes_[0] if len(le.classes_)>0 else '__UNK__')
        vals_safe = vals.where(vals.isin(known), fallback)
    else:
        vals_safe = vals
    return le.transform(vals_safe)

# ---------- Load ----------
print("Loading dataset from:", DATA_PATH)
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
df_raw = pd.read_csv(DATA_PATH, low_memory=False)
print("Rows loaded:", len(df_raw))

# ---------- Detect key columns and normalize names ----------
votes_col = find_col(df_raw, ['votes','votes_obtained','votes_received','valid_votes_gained','Votes'])
valid_votes_col = find_col(df_raw, ['valid_votes','validvotes','valid_votes_polled','Valid_Votes','total_valid_votes','votes_polled'])
pos_col = find_col(df_raw, ['position','pos','rank','result_position'])
pc_name_col = find_col(df_raw, ['pc_name','parliament_constituency','PC_Name','pc'])
pc_no_col = find_col(df_raw, ['pc_no','pc_number','PC_No'])
ac_col = find_col(df_raw, ['constituency','constituency_name','Constituency_Name','AC_NAME','assembly_constituency'])
state_col = find_col(df_raw, ['state','state_name','st_name'])
year_col = find_col(df_raw, ['year','election_year','electionyear'])

print("Detected columns:")
print(" votes_col:", votes_col)
print(" valid_votes_col:", valid_votes_col)
print(" position_col:", pos_col)
print(" pc_name_col:", pc_name_col, "pc_no_col:", pc_no_col, "ac_col:", ac_col)
print(" state_col:", state_col, "year_col:", year_col)

# ---------- Basic normalization ----------
df = df_raw.copy()
df['state_norm'] = df[state_col].astype(str).str.strip().str.lower() if state_col else ""
# PC fields
if pc_name_col:
    df['pc_name'] = df[pc_name_col].astype(str)
else:
    df['pc_name'] = df[ac_col].astype(str) if ac_col else ""
df['pc_name_norm'] = df['pc_name'].astype(str).map(lambda s: str(s).strip().lower())
df['pc_no'] = df[pc_no_col] if pc_no_col else None
# AC fields
if ac_col:
    df['ac_name'] = df[ac_col].astype(str)
    df['ac_name_norm'] = df['ac_name'].astype(str).map(lambda s: str(s).strip().lower())
else:
    df['ac_name'] = ""
    df['ac_name_norm'] = ""
# candidate & party
cand_col = find_col(df, ['candidate','cand_name','name'])
party_col = find_col(df, ['party','party_name'])
df['candidate'] = df[cand_col].astype(str) if cand_col else df.get('candidate','').astype(str)
df['party'] = df[party_col].astype(str) if party_col else df.get('party','').astype(str)
# year
if year_col:
    df['year'] = safe_num(df[year_col]).astype('Int64')
else:
    df['year'] = safe_num(df.get('year', np.nan)).astype('Int64')
# votes_raw
if votes_col:
    df['votes_raw'] = safe_num(df[votes_col]).fillna(0).astype(float)
else:
    df['votes_raw'] = safe_num(df.get('cand_total_votes_acsum', df.get('votes', np.nan))).fillna(0).astype(float)
# valid votes at AC
if valid_votes_col:
    df['valid_votes_ac'] = safe_num(df[valid_votes_col])
else:
    df['valid_votes_ac'] = safe_num(df.get('total_votes', np.nan))
# position at AC (if present)
if pos_col:
    df['position_ac'] = pd.to_numeric(df[pos_col], errors='coerce')
else:
    df['position_ac'] = np.nan
# candidate type and myneta education (if present)
candidate_type_col = find_col(df, ['candidate_type','type','caste','candidate_category','category'])
edu_col = find_col(df, ['myneta_education','education','edu','education_level'])

# ---------- Aggregate AC -> PC candidate totals ----------
print("Aggregating AC -> PC: summing candidate votes across AC segments that belong to same PC...")
group_cols = ['state_norm','pc_name_norm','pc_no','candidate','party','year']
agg = df.groupby(group_cols, dropna=False).agg({'votes_raw':'sum'}).reset_index().rename(columns={'votes_raw':'cand_votes_pc'})
pc_total = df.groupby(['state_norm','pc_name_norm','pc_no','year'], dropna=False).agg({'valid_votes_ac':'sum'}).reset_index().rename(columns={'valid_votes_ac':'pc_total_votes_polled'})
pc_level = pd.merge(agg, pc_total, on=['state_norm','pc_name_norm','pc_no','year'], how='left')
pc_level['vote_share_pc'] = pc_level['cand_votes_pc'] / pc_level['pc_total_votes_polled']
pc_level = pc_level.sort_values(['state_norm','pc_name_norm','pc_no','year','cand_votes_pc'], ascending=[True,True,True,True,False])
pc_level['pc_rank'] = pc_level.groupby(['state_norm','pc_name_norm','pc_no','year'])['cand_votes_pc'].rank(method='dense', ascending=False).astype(int)
pc_level['is_winner_pc'] = (pc_level['pc_rank'] == 1).astype(int)
pc_level['position_pc'] = pc_level['pc_rank']

print("PC-level candidate rows:", len(pc_level))

# ---------- PC-level aggregates (margin etc.) ----------
pc_agg = pc_level.groupby(['state_norm','pc_name_norm','pc_no','year'], dropna=False).agg(
    pc_total_votes_polled=('pc_total_votes_polled','first'),
    top_votes=('cand_votes_pc','max'),
    num_candidates=('candidate','nunique')
).reset_index()

def runner_up(group):
    vals = group.sort_values('cand_votes_pc', ascending=False)['cand_votes_pc'].values
    return vals[1] if len(vals)>1 else np.nan

runner = pc_level.groupby(['state_norm','pc_name_norm','pc_no','year']).apply(runner_up).reset_index().rename(columns={0:'runner_up_votes'})
pc_agg = pd.merge(pc_agg, runner, on=['state_norm','pc_name_norm','pc_no','year'], how='left')

pc_agg['margin_abs'] = pc_agg['top_votes'] - pc_agg['runner_up_votes']
pc_agg['margin_pct'] = pc_agg['margin_abs'] / pc_agg['pc_total_votes_polled']
pc_agg['top_vote_share'] = pc_agg['top_votes'] / pc_agg['pc_total_votes_polled']

pc_agg = pc_agg.sort_values(['state_norm','pc_name_norm','pc_no','year'])
pc_agg['prev_margin_pct'] = pc_agg.groupby(['state_norm','pc_name_norm','pc_no'])['margin_pct'].shift(1)
pc_agg['margin_change_abs'] = (pc_agg['margin_pct'] - pc_agg['prev_margin_pct']).abs()
vol = pc_agg.groupby(['state_norm','pc_name_norm','pc_no'])['margin_change_abs'].mean().reset_index().rename(columns={'margin_change_abs':'margin_volatility'})

pc_level = pd.merge(pc_level, pc_agg[['state_norm','pc_name_norm','pc_no','year','margin_pct','top_vote_share','num_candidates','pc_total_votes_polled']], on=['state_norm','pc_name_norm','pc_no','year'], how='left')
pc_level = pd.merge(pc_level, vol, on=['state_norm','pc_name_norm','pc_no'], how='left')

# ---------- Merge census attributes (c_*) from representative AC row if present ----------
census_cols = [c for c in df_raw.columns if str(c).startswith('c_')]
if census_cols:
    rep = df.groupby(['state_norm','pc_name_norm','pc_no','year'], dropna=False).first().reset_index()
    rep_small = rep[['state_norm','pc_name_norm','pc_no','year'] + census_cols]
    pc_level = pd.merge(pc_level, rep_small, on=['state_norm','pc_name_norm','pc_no','year'], how='left')
    print("Attached census columns:", census_cols[:10])
else:
    print("No c_ prefixed census columns found in original file.")

# Compute turnout change
pc_level.drop('pc_total_votes_polled_y', axis=1, inplace=True, errors='ignore')
pc_level.rename(columns={'pc_total_votes_polled_x':'pc_total_votes_polled'}, inplace=True)
pc_level = pc_level.sort_values(['state_norm','pc_name_norm','pc_no','year'])
pc_level['prev_pc_total_votes_polled'] = pc_level.groupby(['state_norm','pc_name_norm','pc_no'])['pc_total_votes_polled'].shift(1)
pc_level['turnout_change'] = (pc_level['pc_total_votes_polled'] - pc_level['prev_pc_total_votes_polled']) / pc_level['prev_pc_total_votes_polled']

# ---------- Feature engineering ----------
electoral_feats = []
for col in ['cand_votes_pc','vote_share_pc','margin_pct','top_vote_share','num_candidates','pc_total_votes_polled','margin_abs','turnout_change','margin_volatility']:
    if col in pc_level.columns:
        electoral_feats.append(col)

# candidate_type mapping (from df)
if candidate_type_col and candidate_type_col in df.columns:
    cand_type_map = df.groupby(['state_norm','pc_name_norm','pc_no','year','candidate']).first()[candidate_type_col].to_dict()
    def get_cand_type(row):
        k = (row['state_norm'], row['pc_name_norm'], row['pc_no'], row['year'], row['candidate'])
        return cand_type_map.get(k, 'UNKNOWN')
    pc_level['candidate_type'] = pc_level.apply(get_cand_type, axis=1)
else:
    pc_level['candidate_type'] = 'UNKNOWN'

# contested: approximate by count of distinct years per candidate
contested_counts = df.groupby('candidate')['year'].nunique().to_dict()
pc_level['contested'] = pc_level['candidate'].map(lambda x: contested_counts.get(x, 0))

# myneta education mapping
if edu_col and edu_col in df.columns:
    edu_map = df.groupby(['state_norm','pc_name_norm','pc_no','year','candidate']).first()[edu_col].to_dict()
    pc_level['myneta_education'] = pc_level.apply(lambda r: edu_map.get((r['state_norm'], r['pc_name_norm'], r['pc_no'], r['year'], r['candidate']), 'UNKNOWN'), axis=1)
else:
    pc_level['myneta_education'] = 'UNKNOWN'

# Identify urban/rural flag if available as c_Total/Rural/Urban triplet columns
# Some of your earlier steps created c_ columns; if you have c_triplet_col style, use it; else rely on presence of c_Total and c_Rural and c_Urban.
# Here we expect rep_small brought c_ prefixed census columns that include the triplet indicator and population fields.
# If you added an explicit urban flag column earlier (e.g., 'urban_flag'), keep it; otherwise, attempt to compute it:
if 'urban_flag' not in pc_level.columns:
    # Try to compute a simple flag if we have c_total_population_rural and c_total_population_urban or similar
    # We attempt heuristics: find numeric c_* columns that look like total/rural/urban population
    pop_cols = [c for c in pc_level.columns if c.startswith('c_') and ('pop' in c.lower() or 'population' in c.lower())]
    # heuristics might fail; leave urban_flag as NaN if not computable
    if pop_cols:
        # Try to infer which are rural vs urban by name
        urban_cols = [c for c in pop_cols if 'urban' in c.lower()]
        rural_cols = [c for c in pop_cols if 'rural' in c.lower()]
        if urban_cols and rural_cols:
            # compare urban vs rural counts
            pc_level['urban_population'] = pd.to_numeric(pc_level[urban_cols[0]], errors='coerce').fillna(0)
            pc_level['rural_population'] = pd.to_numeric(pc_level[rural_cols[0]], errors='coerce').fillna(0)
            pc_level['urban_flag'] = (pc_level['urban_population'] >= pc_level['rural_population']).astype(int)
        else:
            pc_level['urban_flag'] = np.nan
    else:
        pc_level['urban_flag'] = np.nan

# ---------- Prepare feature lists ----------
census_feats = [c for c in pc_level.columns if str(c).startswith('c_') and pd.api.types.is_numeric_dtype(pc_level[c])]
if 'c_urban_pct' in pc_level.columns:
    census_feats.append('c_urban_pct')

print("Electoral numeric features:", electoral_feats)
print("Census numeric features:", census_feats)

num_features = electoral_feats + census_feats
cat_features = []
if 'candidate_type' in pc_level.columns:
    cat_features.append('candidate_type')
if 'myneta_education' in pc_level.columns:
    cat_features.append('myneta_education')
if 'party' in pc_level.columns:
    cat_features.append('party')

# label
pc_level['winner_label'] = pc_level['is_winner_pc'].astype(int)

# clean numeric
for c in num_features:
    if c in pc_level.columns:
        pc_level[c] = pd.to_numeric(pc_level[c], errors='coerce')
        med = pc_level[c].median(skipna=True)
        pc_level[c] = pc_level[c].fillna(med if not np.isnan(med) else 0.0)

# party OHE
if 'party' in pc_level.columns:
    top_parties = pc_level['party'].value_counts().nlargest(20).index.tolist()
    pc_level['party_top'] = pc_level['party'].where(pc_level['party'].isin(top_parties), 'OTHER')
    party_ohe = pd.get_dummies(pc_level['party_top'], prefix='party')
else:
    party_ohe = pd.DataFrame(index=pc_level.index)

# label encoders
encoders = {}
if 'candidate_type' in pc_level.columns:
    le_ct = LabelEncoder()
    pc_level['candidate_type_le'] = le_ct.fit_transform(safe_label_transform_for_train(pc_level['candidate_type']))
    encoders['candidate_type'] = le_ct
if 'myneta_education' in pc_level.columns:
    le_edu = LabelEncoder()
    pc_level['edu_le'] = le_edu.fit_transform(safe_label_transform_for_train(pc_level['myneta_education']))
    encoders['education'] = le_edu

# final matrices
X_num = pc_level[num_features].reset_index(drop=True) if num_features else pd.DataFrame(index=pc_level.index)
X_cat = pd.concat([party_ohe.reset_index(drop=True),
                   pc_level[['candidate_type_le','edu_le']].reset_index(drop=True) if 'candidate_type_le' in pc_level.columns or 'edu_le' in pc_level.columns else pd.DataFrame(index=pc_level.index),
                  ], axis=1).fillna(0)

X = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1).fillna(0)
y = pc_level['winner_label'].astype(int).reset_index(drop=True)

print("Final feature dimension:", X.shape)

# Train/test split (time-aware)
if 'year' in pc_level.columns and pc_level['year'].notna().any():
    latest_year = int(pc_level['year'].max())
    train_mask = pc_level['year'] < latest_year
    test_mask = pc_level['year'] == latest_year
    if train_mask.sum() >= 50 and test_mask.sum() >= 20:
        X_train = X[train_mask.values]
        y_train = y[train_mask.values]
        X_test = X[test_mask.values]
        y_test = y[test_mask.values]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()>1 else None)

print("Training rows:", len(X_train), "Test rows:", len(X_test))

# scale numeric features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Save preprocessing artifacts
preproc = {
    'scaler': scaler,
    'encoders': encoders,
    'num_features': num_features,
    'cat_columns': list(X_cat.columns),
    'feature_columns': list(X.columns),
    'top_parties': top_parties if 'party' in pc_level.columns else None
}
joblib.dump(preproc, MODELS_DIR / "preprocessing.joblib")
print("Saved preprocessing to:", MODELS_DIR / "preprocessing.joblib")

# ---------- Train models ----------
trained_models = {}
metrics = {}

# Logistic (scaled)
print("Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
lr.fit(X_train_s, y_train)
lr_cal = CalibratedClassifierCV(lr, method='sigmoid', cv='prefit')
lr_cal.fit(X_train_s, y_train)
trained_models['logreg'] = lr_cal
m, _ = eval_model_probs(lr_cal, X_test_s, y_test)
metrics['logreg'] = m
print("LogReg:", m)

# RandomForest (raw)
print("Training RandomForest...")
rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
rf_cal = CalibratedClassifierCV(rf, method='sigmoid', cv=5)
rf_cal.fit(X_train, y_train)
trained_models['random_forest'] = rf_cal
m, _ = eval_model_probs(rf_cal, X_test, y_test)
metrics['random_forest'] = m
print("RF:", m)

# XGBoost
if xgb is not None:
    try:
        print("Training XGBoost...")
        xgb_clf = xgb.XGBClassifier(n_estimators=300, use_label_encoder=False, eval_metric='logloss', n_jobs=-1, random_state=42)
        xgb_clf.fit(X_train, y_train)
        trained_models['xgboost'] = xgb_clf
        m,_ = eval_model_probs(xgb_clf, X_test, y_test)
        metrics['xgboost'] = m
        print("XGBoost:", m)
    except Exception as e:
        print("XGBoost training failed:", e)

# MLP (scaled)
print("Training MLP...")
mlp = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=400, random_state=42)
mlp.fit(X_train_s, y_train)
trained_models['mlp'] = mlp
m,_ = eval_model_probs(mlp, X_test_s, y_test)
metrics['mlp'] = m
print("MLP:", m)

# TabNet optional
# ===== Safe TabNet training block (replace your previous TabNet code) =====
if TABNET_AVAILABLE:
    # robust tabnet training snippet
    try:
        # prefer the installed TabNet import (either package)
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier

            tb_name = "pytorch-tabnet"
        except Exception:
            from eh_pytorch_tabnet.tab_model import TabNetClassifier

            tb_name = "eh-pytorch-tabnet"
        print("TabNet package:", tb_name)

        import torch

        use_gpu = torch.cuda.is_available()
        device = "cuda" if use_gpu else "cpu"
        print("TabNet training device:", device)

        # prototype on small subset to test call signature and API
        n_proto = min(200, X_train.shape[0])
        X_proto = X_train.values[:n_proto]
        y_proto = y_train.values[:n_proto]
        X_val_proto = X_test.values[:min(200, X_test.shape[0])]
        y_val_proto = y_test.values[:min(200, y_test.shape[0])]

        # Construct safe kwargs (avoid 'verbose' if not supported)
        tb_kwargs = dict(
            seed=42,
            optimizer_params=dict(lr=2e-2),
        )
        # Use device_name param only if signature accepts it
        import inspect

        sig = inspect.signature(TabNetClassifier.__init__)
        if 'device_name' in sig.parameters:
            tb_kwargs['device_name'] = device

        tabnet = TabNetClassifier(**tb_kwargs)

        # Fit with safe args; many TabNet versions accept max_epochs, patience, batch_size
        fit_kwargs = dict(
            X_train=X_proto, y_train=y_proto,
            eval_set=[(X_val_proto, y_val_proto)],
            max_epochs=50,
            patience=10,
            batch_size=256,
            virtual_batch_size=64,
            drop_last=False
        )
        # Some TabNet accepts numpy arrays only — ensure numpy:
        for k in ('X_train', 'y_train'):
            if isinstance(fit_kwargs[k], pd.DataFrame) or isinstance(fit_kwargs[k], pd.Series):
                fit_kwargs[k] = fit_kwargs[k].values

        # remove keys not supported by the installed TabNet.fit()
        fit_sig = inspect.signature(tabnet.fit)
        fit_kwargs = {k: v for k, v in fit_kwargs.items() if k in fit_sig.parameters}

        print("Attempting TabNet prototyping (small subset)...")
        tabnet.fit(**fit_kwargs)
        print("TabNet prototype training successful. Now train on full data (this may take time).")

        # Now train on full data (optionally with same fit kwargs but with full arrays)
        fit_kwargs['X_train'] = X_train.values if hasattr(X_train, 'values') else X_train
        fit_kwargs['y_train'] = y_train.values if hasattr(y_train, 'values') else y_train
        fit_kwargs['eval_set'] = [(X_test.values if hasattr(X_test, 'values') else X_test,
                                   y_test.values if hasattr(y_test, 'values') else y_test)]
        tabnet.fit(**{k: v for k, v in fit_kwargs.items() if k in fit_sig.parameters})

        trained_models['tabnet'] = tabnet

    except Exception as e:
        print("TabNet training failed:", type(e).__name__, e)
        print("Falling back to HistGradientBoostingClassifier (sklearn) or other models.")
        from sklearn.ensemble import HistGradientBoostingClassifier

        hgb = HistGradientBoostingClassifier(max_iter=200, random_state=42)
        hgb.fit(X_train, y_train)
        trained_models['hist_gb_fallback'] = hgb
# ========================================================================


# LightGBM
# ===== Robust LightGBM training (sanitized column names approach) =====
try:
    print("Training LightGBM (robust call)...")
    # Use DataFrame with sanitized column names to avoid whitespace warnings
    def sanitize_colnames(df):
        new_cols = []
        for c in df.columns:
            if isinstance(c, str):
                nc = c.strip().replace(' ', '_').replace('/', '_').replace('%','pct').replace('-','_')
                nc = re.sub(r'[^0-9a-zA-Z_]', '', nc)
                new_cols.append(nc)
            else:
                new_cols.append(str(c))
        df.columns = new_cols
        return df

    X_train_lgb = sanitize_colnames(X_train.copy())
    X_test_lgb  = sanitize_colnames(X_test.copy())

    # Align columns
    for c in list(X_train_lgb.columns):
        if c not in X_test_lgb.columns:
            X_test_lgb[c] = 0.0
    extra = [c for c in X_test_lgb.columns if c not in X_train_lgb.columns]
    if extra:
        X_test_lgb = X_test_lgb.drop(columns=extra)
    X_test_lgb = X_test_lgb[X_train_lgb.columns]

    lgb_clf = lgb.LGBMClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    # Use eval_metric and early stopping; DO NOT pass 'verbose' here for compatibility
    lgb_clf.fit(X_train_lgb, y_train,
                eval_set=[(X_test_lgb, y_test)],
                eval_metric='auc',
                early_stopping_rounds=20)
    trained_models['lightgbm'] = lgb_clf
    m,_ = eval_model_probs(lgb_clf, X_test_lgb, y_test)
    metrics['lightgbm'] = m
    print("LightGBM:", m)

except TypeError as e:
    # older/newer versions mismatch — try fallback using numpy arrays (no named eval kwargs)
    print("LightGBM fit raised TypeError (version mismatch). Retrying with numpy arrays without fit kwargs:", e)
    try:
        lgb_clf = lgb.LGBMClassifier(n_estimators=300, n_jobs=-1, random_state=42)
        lgb_clf.fit(X_train.values, y_train)
        trained_models['lightgbm'] = lgb_clf
        m,_ = eval_model_probs(lgb_clf, X_test, y_test)
        metrics['lightgbm'] = m
        print("LightGBM (numpy) metrics:", m)
    except Exception as e2:
        print("LightGBM training failed (numpy attempt):", type(e2).__name__, e2)
        print("Falling back to sklearn.HistGradientBoostingClassifier.")
        from sklearn.ensemble import HistGradientBoostingClassifier
        hgb = HistGradientBoostingClassifier(max_iter=300, random_state=42)
        hgb.fit(X_train, y_train)
        trained_models['hist_gb_fallback'] = hgb
        m,_ = eval_model_probs(hgb, X_test, y_test)
        metrics['hist_gb_fallback'] = m
        print("Fallback metrics:", m)
# =====================================================================

if lgb is not None:
    try:
        print("Training LightGBM...")
        lgb_clf = lgb.LGBMClassifier(n_estimators=300, n_jobs=-1, random_state=42)
        lgb_clf.fit(X_train, y_train)
        trained_models['lightgbm'] = lgb_clf
        m,_ = eval_model_probs(lgb_clf, X_test, y_test)
        metrics['lightgbm'] = m
        print("LightGBM:", m)
    except Exception as e:
        print("LightGBM training failed:", e)

# Save trained models
for name, model in trained_models.items():
    joblib.dump({'model': model, 'feature_columns': list(X.columns)}, MODELS_DIR / f"{name}.joblib")
    print("Saved model:", MODELS_DIR / f"{name}.joblib")

# Save metrics
with open(OUT_DIR / "model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)
print("Saved model metrics to:", OUT_DIR / "model_metrics.json")

# ---------- Predictions for latest year and ensemble ----------
print("Preparing predictions for latest year / full ensemble...")
if 'year' in pc_level.columns and pc_level['year'].notna().any():
    latest_year = int(pc_level['year'].max())
    df_latest = pc_level[pc_level['year'] == latest_year].copy()
else:
    df_latest = pc_level.copy()
df_latest = df_latest.reset_index(drop=True)

# Rebuild X_latest aligned to training features using saved preprocessing
pre = joblib.load(MODELS_DIR / "preprocessing.joblib")
train_feature_cols = pre.get('feature_columns')
train_num_features = pre.get('num_features', [])
train_cat_columns = pre.get('cat_columns', [])
encoders = pre.get('encoders', {})
top_parties = pre.get('top_parties', None)
scaler = pre.get('scaler', None)

# numeric part
X_latest_num = pd.DataFrame(index=df_latest.index)
for c in train_num_features:
    if c in df_latest.columns:
        X_latest_num[c] = pd.to_numeric(df_latest[c], errors='coerce').fillna(0)
    else:
        X_latest_num[c] = 0.0

# categorical part
X_latest_cat = pd.DataFrame(index=df_latest.index)
# party OHE
if 'party' in df_latest.columns:
    if top_parties is not None:
        df_latest['party_top'] = df_latest['party'].where(df_latest['party'].isin(top_parties), 'OTHER')
    else:
        df_latest['party_top'] = df_latest['party'].fillna('OTHER')
    party_ohe_latest = pd.get_dummies(df_latest['party_top'], prefix='party')
else:
    party_ohe_latest = pd.DataFrame(index=df_latest.index)
# align party OHE to train's party columns
party_ohe_aligned = party_ohe_latest.reindex(columns=[c for c in train_cat_columns if c.startswith('party_')], fill_value=0)
X_latest_cat = pd.concat([X_latest_cat, party_ohe_aligned.reset_index(drop=True)], axis=1)

# label-encoded features
if 'candidate_type' in df_latest.columns and 'candidate_type' in encoders:
    le_ct = encoders['candidate_type']
    vals = df_latest['candidate_type'].astype(str).fillna('NA')
    transformed = safe_label_transform_for_apply(le_ct, vals)
    X_latest_cat['candidate_type_le'] = pd.Series(transformed, index=df_latest.index)
elif 'candidate_type_le' in train_feature_cols:
    X_latest_cat['candidate_type_le'] = 0

if 'myneta_education' in df_latest.columns and 'education' in encoders:
    le_edu = encoders['education']
    vals = df_latest['myneta_education'].astype(str).fillna('NA')
    transformed = safe_label_transform_for_apply(le_edu, vals)
    X_latest_cat['edu_le'] = pd.Series(transformed, index=df_latest.index)
elif 'edu_le' in train_feature_cols:
    X_latest_cat['edu_le'] = 0

# ensure other categorical columns from training exist
for c in train_cat_columns:
    if c not in X_latest_cat.columns:
        X_latest_cat[c] = 0

# combine and align to train_feature_cols
X_latest_full = pd.concat([X_latest_num.reset_index(drop=True), X_latest_cat.reset_index(drop=True)], axis=1).fillna(0)
for c in train_feature_cols:
    if c not in X_latest_full.columns:
        X_latest_full[c] = 0.0
# drop extras
extra_cols = [c for c in X_latest_full.columns if c not in train_feature_cols]
if extra_cols:
    X_latest_full = X_latest_full.drop(columns=extra_cols)
# reorder
X_latest_full = X_latest_full[train_feature_cols]

# scale
if scaler is None:
    raise RuntimeError("Scaler missing in preprocessing.joblib")
X_latest_s = scaler.transform(X_latest_full)

# assign final X_latest
X_latest = X_latest_full.copy()

# predict with each model
prob_cols = []
for name, model in trained_models.items():
    col = f'prob_{name}'
    prob_cols.append(col)
    try:
        if name in ['logreg','mlp']:
            probs = model.predict_proba(X_latest_s)[:,1] if hasattr(model, "predict_proba") else model.predict(X_latest_s)
        else:
            probs = model.predict_proba(X_latest)[:,1] if hasattr(model, "predict_proba") else model.predict(X_latest)
        df_latest[col] = probs
    except Exception:
        # fallback attempt with numpy arrays
        try:
            if name in ['logreg','mlp']:
                df_latest[col] = model.predict_proba(X_latest_s.values)[:,1]
            else:
                df_latest[col] = model.predict_proba(X_latest.values)[:,1]
        except Exception:
            df_latest[col] = 0.0

# ensemble average
prob_cols_present = [c for c in prob_cols if c in df_latest.columns]
if prob_cols_present:
    df_latest['avg_prob'] = df_latest[prob_cols_present].mean(axis=1)
else:
    df_latest['avg_prob'] = 0.0

# recommended candidate per PC
recs = df_latest.groupby(['state_norm','pc_name_norm','pc_no']).apply(
    lambda g: g.loc[g['avg_prob'].idxmax(), ['state_norm','pc_name_norm','pc_no','candidate','party','avg_prob']]
).reset_index(drop=True)
recs_path = OUT_DIR / "recommended_candidates_pc.csv"
recs.to_csv(recs_path, index=False)
print("Saved recommended candidates to:", recs_path)

df_latest.to_csv(OUT_DIR / "df_latest_predictions.csv", index=False)

# swing ranking
def pc_closeness(group):
    probs = group['avg_prob'].values
    if len(probs) <= 1:
        return 1.0
    s = np.sort(probs)[::-1]
    return float(s[0] - s[1])

pc_scores = df_latest.groupby(['state_norm','pc_name_norm','pc_no']).apply(lambda g: pd.Series({
    'pc_margin_pct': g['margin_pct'].iloc[0] if 'margin_pct' in g.columns else np.nan,
    'pc_margin_volatility': g['margin_volatility'].iloc[0] if 'margin_volatility' in g.columns else np.nan,
    'closeness': pc_closeness(g)
})).reset_index()

def norm01(x):
    x = np.array(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mn == mx:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)

pc_scores['pc_margin_pct_f'] = pc_scores['pc_margin_pct'].fillna(pc_scores['pc_margin_pct'].median())
pc_scores['pc_margin_volatility_f'] = pc_scores['pc_margin_volatility'].fillna(pc_scores['pc_margin_volatility'].median())
pc_scores['closeness_f'] = pc_scores['closeness'].fillna(pc_scores['closeness'].median())
pc_scores['swing_score'] = 0.4*(1 - norm01(pc_scores['pc_margin_pct_f'])) + 0.4*norm01(pc_scores['pc_margin_volatility_f']) + 0.2*norm01(1 - pc_scores['closeness_f'])
pc_scores = pc_scores.sort_values('swing_score', ascending=False).reset_index(drop=True)
pc_scores.to_csv(OUT_DIR / "pc_swing_ranking.csv", index=False)
print("Saved PC swing ranking to:", OUT_DIR / "pc_swing_ranking.csv")

# Save model metrics summary already created earlier
print("DONE. Outputs written to:", OUT_DIR)
print("Files of interest:")
for f in ["recommended_candidates_pc.csv", "pc_swing_ranking.csv", "df_latest_predictions.csv", "model_metrics.json"]:
    p = OUT_DIR / f
    print("-", p, "(exists)" if p.exists() else "(MISSING)")
