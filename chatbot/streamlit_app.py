# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
from typing import Dict, Any

st.set_page_config(layout="wide", page_title="Election ML Dashboard")

# --------- CONFIG: adjust these paths if your files are elsewhere ----------
BASE_DIR = Path("./training/election_outputs")
MODELS_DIR = BASE_DIR / "models"
RECS_CSV = BASE_DIR / "recommended_candidates_pc.csv"
SWING_CSV = BASE_DIR / "pc_swing_ranking.csv"
PRED_CSV = BASE_DIR / "df_latest_predictions.csv"
METRICS_JSON = BASE_DIR / "model_metrics.json"
# ------------------------------------------------------------------------

st.title("Election Models — Dashboard")

# --------- Utilities ----------
def safe_load_csv(p: Path):
    if p.exists():
        try:
            return pd.read_csv(p, low_memory=False)
        except Exception as e:
            st.error(f"Failed to load CSV {p}: {e}")
            return None
    else:
        st.warning(f"File not found: {p}")
        return None

@st.cache_data(show_spinner=False)
def load_models_and_preproc(models_dir: Path):
    """Load all .joblib models and preprocessing.joblib from models_dir."""
    models = {}
    preproc = None
    if not models_dir.exists():
        st.error(f"Models directory not found: {models_dir}")
        return models, preproc

    for p in models_dir.glob("*.joblib"):
        try:
            obj = joblib.load(p)
            if isinstance(obj, dict) and 'model' in obj:
                models[p.stem] = obj
            else:
                models[p.stem] = {
                    'model': obj,
                    'feature_columns': obj.feature_names_in_ if hasattr(obj, 'feature_names_in_') else None
                }
        except Exception as e:
            st.warning(f"Failed to load {p.name}: {e}")

    preproc_path = models_dir / "preprocessing.joblib"
    if preproc_path.exists():
        try:
            preproc = joblib.load(preproc_path)
        except Exception as e:
            st.warning(f"Could not load preprocessing.joblib: {e}")
    return models, preproc

def align_features(df_row: pd.DataFrame, preproc: Dict[str, Any]):
    """Align to training feature order."""
    train_cols = preproc.get('feature_columns')
    num_features = preproc.get('num_features', [])
    cat_cols = preproc.get('cat_columns', [])
    scaler = preproc.get('scaler', None)
    encoders = preproc.get('encoders', {})

    # numeric
    X_num = pd.DataFrame(index=df_row.index)
    for c in num_features:
        X_num[c] = pd.to_numeric(df_row[c], errors='coerce').fillna(0) if c in df_row else 0.0

    # categorical
    X_cat = pd.DataFrame(index=df_row.index)
    party_ohe_cols = [c for c in cat_cols if c.startswith('party_')]

    if 'party' in df_row.columns:
        party_vals = df_row['party'].fillna('OTHER')
        party_ohe = pd.get_dummies(party_vals, prefix='party')
        party_ohe = party_ohe.reindex(columns=party_ohe_cols, fill_value=0)
        X_cat = pd.concat([X_cat, party_ohe], axis=1)
    else:
        for c in party_ohe_cols:
            X_cat[c] = 0

    # label encoded columns
    if 'candidate_type' in df_row.columns and 'candidate_type' in encoders:
        le = encoders['candidate_type']
        vals = df_row['candidate_type'].astype(str).fillna("NA")
        known = set(le.classes_)
        fallback = "NA" if "NA" in known else list(le.classes_)[0]
        vals = vals.where(vals.isin(known), fallback)
        X_cat['candidate_type_le'] = le.transform(vals)
    elif 'candidate_type_le' in train_cols:
        X_cat['candidate_type_le'] = 0

    if 'myneta_education' in df_row.columns and 'education' in encoders:
        le2 = encoders['education']
        vals = df_row['myneta_education'].astype(str).fillna("NA")
        known = set(le2.classes_)
        fallback = "NA" if "NA" in known else list(le2.classes_)[0]
        vals = vals.where(vals.isin(known), fallback)
        X_cat['edu_le'] = le2.transform(vals)
    elif 'edu_le' in train_cols:
        X_cat['edu_le'] = 0

    # ensure all categorical training cols exist
    for c in cat_cols:
        if c not in X_cat:
            X_cat[c] = 0

    # combine
    X_full = pd.concat([X_num, X_cat], axis=1).fillna(0)

    # ensure all train_cols exist & reorder
    for c in train_cols:
        if c not in X_full:
            X_full[c] = 0.0
    extra = [c for c in X_full if c not in train_cols]
    if extra:
        X_full = X_full.drop(columns=extra)

    X_full = X_full[train_cols]

    # scale
    X_scaled = scaler.transform(X_full) if scaler is not None else None
    return X_full, X_scaled


# --------- Load data & models ----------
with st.spinner("Loading CSVs and models..."):
    recs_df = safe_load_csv(RECS_CSV)
    swing_df = safe_load_csv(SWING_CSV)
    preds_df = safe_load_csv(PRED_CSV)
    metrics = json.load(open(METRICS_JSON)) if METRICS_JSON.exists() else {}
    models, preproc = load_models_and_preproc(MODELS_DIR)

# --------- Main Tabs (Chatbot removed) ----------
tab = st.tabs([
    "Overview",
    "Lookup / Query",
    "Constituency Explorer",
    "Simulator"
])

# ---- Overview ----
with tab[0]:
    st.header("Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Models found", len(models))
    c2.metric("Recommended Parliamentary Constituencies", len(recs_df))
    c3.metric("Parliamentary Constituencies with Predictions", preds_df['pc_name_norm'].nunique())

    st.subheader("Model Metrics")

    if not metrics:
        st.warning("No model metrics found.")
    else:
        # metrics is expected as: { "logreg": { "accuracy":.., "precision":.. }, "random_forest": {...}, ... }
        # Convert to table where each row is a model and columns are metric names
        try:
            # Build a clean metrics table with fixed columns
            rows = []
            for model_name, m in metrics.items():
                row = {
                    "model": model_name,
                    "accuracy": m.get("accuracy"),
                    "precision": m.get("precision"),
                    "recall": m.get("recall"),
                    "f1_score": m.get("f1_score") or m.get("f1-score") or m.get("f1"),
                    "roc_auc": m.get("roc_auc") or m.get("roc-auc") or m.get("roc_auc_score")
                }
                rows.append(row)

            df_metrics = pd.DataFrame(rows, columns=[
                "model", "accuracy", "precision", "recall", "f1_score", "roc_auc"
            ])

            # Round float columns
            for c in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
                if c in df_metrics.columns and pd.api.types.is_float_dtype(df_metrics[c]):
                    df_metrics[c] = df_metrics[c].round(4)

        except Exception:
            # fallback: try building manually
            rows = []
            for mname, mm in metrics.items():
                row = {'model': mname}
                if isinstance(mm, dict):
                    for k, v in mm.items():
                        if isinstance(v, float):
                            row[k] = round(v, 4)
                        else:
                            row[k] = v
                rows.append(row)
            df_metrics = pd.DataFrame(rows)

        st.dataframe(df_metrics)

        # Download button: export metrics table as CSV
        csv = df_metrics.to_csv(index=False)
        st.download_button("Download metrics CSV", csv, file_name="model_metrics_table.csv", mime="text/csv")

    st.subheader("Top Swing PCs")
    st.dataframe(swing_df.head(20))

    st.subheader("Recommended Candidates")
    st.dataframe(recs_df.head(40))


# ---- Lookup / Query ----
with tab[1]:
    st.header("Lookup / Query")
    qtype = st.selectbox("Choose Query", [
        "Candidate by PC",
        "PC Summary",
        "Model per-candidate probabilities"
    ])

    if qtype == "Candidate by PC":
        state = st.selectbox("State", sorted(preds_df['state_norm'].unique()))
        pcs = preds_df[preds_df['state_norm']==state]['pc_name_norm'].unique()
        pc = st.selectbox("PC", pcs)
        if st.button("Show"):
            st.dataframe(recs_df[(recs_df['state_norm']==state)&(recs_df['pc_name_norm']==pc)])

    elif qtype == "PC Summary":
        pc = st.text_input("Enter PC name (normalized):")
        if st.button("Search"):
            st.dataframe(preds_df[preds_df['pc_name_norm']==pc])

    elif qtype == "Model per-candidate probabilities":
        state = st.selectbox("State", sorted(preds_df['state_norm'].unique()))
        pc = st.selectbox("PC", sorted(preds_df[preds_df['state_norm']==state]['pc_name_norm'].unique()))
        if st.button("Show"):
            rows = preds_df[(preds_df['state_norm']==state) & (preds_df['pc_name_norm']==pc)]
            prob_cols = [c for c in rows.columns if c.startswith("prob_") or c=="avg_prob"]
            st.dataframe(rows[['candidate','party'] + prob_cols].sort_values(prob_cols[-1], ascending=False))


# ---- Constituency Explorer ----
with tab[2]:
    st.header("Explore Constituencies")

    state = st.selectbox("State", ["ALL"] + sorted(preds_df['state_norm'].unique()))
    df2 = preds_df if state=="ALL" else preds_df[preds_df['state_norm']==state]

    pc = st.selectbox("PC", ["ALL"] + sorted(df2['pc_name_norm'].unique()))
    if pc != "ALL":
        df2 = df2[df2['pc_name_norm']==pc]

    st.dataframe(df2.sort_values("avg_prob", ascending=False).head(200))


# ---- Simulator ----
with tab[3]:
    st.header("Scenario Simulator")
    if preproc is None:
        st.warning("Preprocessing missing. Cannot simulate.")
    else:
        row_idx = st.number_input("Select row index", min_value=0, max_value=len(preds_df)-1, value=0)
        base = preds_df.loc[[row_idx]].copy()
        st.dataframe(base.reset_index(drop=True))

        st.subheader("Tweaks")
        new_vote_share = st.number_input("vote_share_pc", value=float(base["vote_share_pc"].iloc[0]))
        new_turnout = st.number_input("pc_total_votes_polled", value=float(base["pc_total_votes_polled"].iloc[0]))
        new_urban = st.selectbox("urban_flag", [0,1], index=int(base["urban_flag"].iloc[0] if "urban_flag" in base else 0))

        if st.button("Simulate"):
            sim = base.copy()
            sim["vote_share_pc"] = new_vote_share
            sim["pc_total_votes_polled"] = new_turnout
            if "urban_flag" in sim:
                sim["urban_flag"] = new_urban

            X_full, X_scaled = align_features(sim, preproc)

            results = {}
            for name, obj in models.items():
                m = obj["model"]
                try:
                    if name in ["logreg","mlp"]:
                        prob = m.predict_proba(X_scaled)[0,1]
                    else:
                        prob = m.predict_proba(X_full)[0,1]
                except:
                    prob = None
                results[name] = prob

            st.json(results)
            numeric = [p for p in results.values() if isinstance(p, (int,float))]
            if numeric:
                st.metric("Ensemble Avg", float(np.mean(numeric)))
