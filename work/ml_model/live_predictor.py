import os
import joblib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

from model import resolve_data_path, evaluate_metrics

BUNDLE_FILENAME = "champion_live_bundle.pkl"
TEST_CSV_FILENAME = "unseen_test_dataset_6163.csv"


def get_outputs_dir():
    """Resolve the work/outputs directory path."""
    candidates = [
        os.path.join("work", "outputs"),
        os.path.join("..", "outputs"),
        "outputs",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    out_dir = os.path.join("work", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_bundle_path():
    return os.path.join(get_outputs_dir(), BUNDLE_FILENAME)


def get_test_csv_path():
    return os.path.join(get_outputs_dir(), TEST_CSV_FILENAME)


def derive_tiers(raw: dict) -> dict:
    """Automatically derive categorical tier columns from raw numeric SEO inputs."""
    d = dict(raw)

    # 1. freshness_tier from days_since_last_update
    if not d.get("freshness_tier"):
        days_upd = float(d.get("days_since_last_update", 90))
        if days_upd <= 30:
            d["freshness_tier"] = "0-30"
        elif days_upd <= 90:
            d["freshness_tier"] = "31-90"
        elif days_upd <= 180:
            d["freshness_tier"] = "91-180"
        else:
            d["freshness_tier"] = "181+"

    # 2. position_tier from avg_position
    if not d.get("position_tier"):
        pos = float(d.get("avg_position", 10.0))
        if pos <= 3.0:
            d["position_tier"] = "top_3"
        elif pos <= 10.0:
            d["position_tier"] = "page_1"
        elif pos <= 20.0:
            d["position_tier"] = "striking"
        elif pos <= 50.0:
            d["position_tier"] = "page_3_5"
        else:
            d["position_tier"] = "deep"

    # 3. competition_level from competition
    if not d.get("competition_level"):
        comp = d.get("competition", None)
        if comp is None or (isinstance(comp, float) and np.isnan(comp)):
            d["competition_level"] = "unknown"
        else:
            comp = float(comp)
            if comp < 0.33:
                d["competition_level"] = "LOW"
            elif comp <= 0.66:
                d["competition_level"] = "MEDIUM"
            else:
                d["competition_level"] = "HIGH"

    # 4. age_tier_order from content_age_days
    if not d.get("age_tier_order"):
        age = float(d.get("content_age_days", 180))
        if age <= 90:
            d["age_tier_order"] = 3
        elif age <= 180:
            d["age_tier_order"] = 4
        elif age <= 365:
            d["age_tier_order"] = 5
        else:
            d["age_tier_order"] = 6

    return d


def engineer_features_dataframe(df: pd.DataFrame, reference_feature_cols=None) -> pd.DataFrame:
    """Run the exact 48-signal domain feature engineering pipeline from model.py."""
    df = df.copy()

    # SERP Rank Inverse Power Law Features
    df['has_rank_data'] = (df['avg_position'] > 0).astype(int)
    df['serp_rank_power'] = np.where(df['avg_position'] > 0, 1.0 / df['avg_position'], 0.0)
    df['imputed_avg_position'] = np.where(df['avg_position'] > 0, df['avg_position'], 100.0)
    df['log_avg_position'] = np.log(df['imputed_avg_position'])

    # Missingness Indicator Flags
    df['missing_word_count'] = df['word_count'].isna().astype(int)
    df['missing_search_volume'] = df['search_volume'].isna().astype(int)
    df['missing_competition'] = df['competition'].isna().astype(int)
    df['missing_cpc'] = df['cpc'].isna().astype(int)

    # Behavioral & Engagement Ratios
    df['engaged_session_ratio'] = df['engaged_sessions_90d'] / (df['sessions_90d'] + 1.0)
    df['scroll_per_pageview'] = df['scroll_events_90d'] / (df['pageviews_90d'] + 1.0)
    df['ai_session_ratio'] = df['ai_sessions_90d'] / (df['sessions_90d'] + 1.0)

    # Log Transforms on Skewed Count Variables
    skewed_cols = [
        'word_count', 'char_count', 'search_volume', 'cpc', 'content_age_days',
        'days_since_last_update', 'pageviews_90d', 'sessions_90d', 'users_90d',
        'engaged_sessions_90d', 'ai_sessions_90d', 'scroll_events_90d', 'days_with_sessions'
    ]
    for col in skewed_cols:
        df['log_' + col] = np.log1p(df[col].fillna(0))

    # Interaction Synergies
    df['demand_position_synergy'] = df['log_search_volume'] * df['serp_rank_power']
    df['engagement_volume_synergy'] = df['log_sessions_90d'] * df['serp_rank_power']
    df['depth_engagement_synergy'] = df['log_word_count'] * df['engaged_session_ratio']

    numeric_cols = [
        'has_rank_data', 'serp_rank_power', 'log_avg_position', 'imputed_avg_position',
        'missing_word_count', 'missing_search_volume', 'missing_competition', 'missing_cpc',
        'competition', 'age_tier_order', 'engagement_rate', 'scroll_rate', 'ai_traffic_pct',
        'engaged_session_ratio', 'scroll_per_pageview', 'ai_session_ratio',
        'demand_position_synergy', 'engagement_volume_synergy', 'depth_engagement_synergy'
    ] + ['log_' + c for c in skewed_cols]

    cat_cols = ['content_type', 'main_intent', 'competition_level', 'freshness_tier', 'position_tier']
    dummy_dfs = [
        pd.get_dummies(df[c].fillna('unknown'), prefix=c, drop_first=(reference_feature_cols is None), dtype=float)
        for c in cat_cols
    ]

    df_features = pd.concat([df[numeric_cols]] + dummy_dfs, axis=1)
    df_features = df_features.loc[:, ~df_features.columns.duplicated()]

    if reference_feature_cols is not None:
        for col in reference_feature_cols:
            if col not in df_features.columns:
                df_features[col] = 0.0
        df_features = df_features[reference_feature_cols]

    return df_features


def train_and_save_bundle(force_retrain=False):
    """Train the models from model.py once, export the 6,163 held-out test set CSV, and cache the bundle."""
    bundle_path = get_bundle_path()
    test_csv_path = get_test_csv_path()

    if os.path.exists(bundle_path) and os.path.exists(test_csv_path) and not force_retrain:
        return joblib.load(bundle_path)

    data_path = resolve_data_path()
    df = pd.read_csv(data_path)
    df['target_log_impressions'] = np.log1p(df['impressions_90d'])

    X_all = engineer_features_dataframe(df, reference_feature_cols=None)
    feature_cols = list(X_all.columns)
    y_all = df['target_log_impressions']
    groups = df['client_id']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X_all, y_all, groups=groups))

    X_train, X_test = X_all.iloc[train_idx].copy(), X_all.iloc[test_idx].copy()
    y_train, y_test = y_all.iloc[train_idx].copy(), y_all.iloc[test_idx].copy()

    # Export the 6,163 unseen held-out test rows for live testing & exploration
    df_test_raw = df.iloc[test_idx].copy().reset_index(drop=True)

    # Baseline 1: Mean Dummy
    y_train_mean = float(y_train.mean())

    # Baseline 2: Univariate Word Count OLS
    lr_univariate = LinearRegression().fit(X_train[['log_word_count']], y_train)

    # Candidate A: Polynomial Interaction Ridge
    interaction_signals = [
        'log_sessions_90d', 'log_pageviews_90d', 'serp_rank_power', 'log_avg_position',
        'log_word_count', 'log_search_volume', 'log_days_since_last_update',
        'engaged_session_ratio', 'log_users_90d', 'log_engaged_sessions_90d', 'has_rank_data',
        'demand_position_synergy', 'engagement_volume_synergy'
    ]
    poly_ridge_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)),
        ('scaler', StandardScaler()),
        ('regressor', RidgeCV(alphas=np.logspace(-1, 5, 30), cv=5))
    ])
    poly_ridge_pipe.fit(X_train[interaction_signals], y_train)
    best_alpha = float(poly_ridge_pipe.named_steps['regressor'].alpha_)

    # Candidate B: HistGradientBoostingRegressor
    hgbr_model = HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.03,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=2.0,
        random_state=42
    )
    hgbr_model.fit(X_train, y_train)

    # Evaluate on held-out test set for benchmark display
    y_pred_dummy = np.full_like(y_test, y_train_mean)
    y_pred_uni = lr_univariate.predict(X_test[['log_word_count']])
    y_pred_ridge = poly_ridge_pipe.predict(X_test[interaction_signals])
    y_pred_hgbr = hgbr_model.predict(X_test)
    y_pred_ens = 0.50 * y_pred_ridge + 0.50 * y_pred_hgbr

    df_test_raw["pred_log_impressions"] = np.round(y_pred_ens, 4)
    df_test_raw["pred_impressions_90d"] = np.round(np.expm1(np.maximum(0.0, y_pred_ens))).astype(int)
    df_test_raw["opportunity_gap_log"] = np.round(y_pred_ens - y_test.values, 4)
    df_test_raw.to_csv(test_csv_path, index=False)

    benchmark_table = [
        evaluate_metrics(y_test, y_pred_dummy, "Baseline 1: Mean Dummy Regressor"),
        evaluate_metrics(y_test, y_pred_uni, "Baseline 2: Univariate Rule (log_word_count)"),
        evaluate_metrics(y_test, y_pred_ridge, f"Candidate A: Poly Ridge (alpha={best_alpha:.2f})"),
        evaluate_metrics(y_test, y_pred_hgbr, "Candidate B: HistGradientBoosting"),
        evaluate_metrics(y_test, y_pred_ens, "Champion: Stacking Ensemble (Ridge + HGBR)"),
    ]

    bundle = {
        "feature_cols": feature_cols,
        "interaction_signals": interaction_signals,
        "y_train_mean": y_train_mean,
        "lr_univariate": lr_univariate,
        "poly_ridge_pipe": poly_ridge_pipe,
        "hgbr_model": hgbr_model,
        "best_alpha": best_alpha,
        "benchmark_table": benchmark_table,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_clients": int(groups.iloc[train_idx].nunique()),
        "test_clients": int(groups.iloc[test_idx].nunique()),
    }
    joblib.dump(bundle, bundle_path)
    return bundle


def load_unseen_test_dataframe() -> pd.DataFrame:
    """Load the 6,163 held-out unseen client rows exported during bundle creation."""
    test_csv_path = get_test_csv_path()
    if not os.path.exists(test_csv_path):
        train_and_save_bundle(force_retrain=True)
    return pd.read_csv(test_csv_path)


def row_to_preset_dict(row: pd.Series) -> dict:
    """Convert any raw row from unseen_test_dataset_6163.csv into a clean input dict for the UI."""
    def clean_num(val, default=0.0):
        if pd.isna(val):
            return default
        return float(val)

    wc = int(clean_num(row.get("word_count"), 1500))
    cc = int(clean_num(row.get("char_count"), wc * 6))
    return {
        "description": (
            f"Held-Out Unseen Test Page `{row.get('content_id', 'unknown')}` "
            f"(Actual 90d Impressions = {int(clean_num(row.get('impressions_90d'), 0)):,})."
        ),
        "content_type": str(row.get("content_type", "keyword article")) if pd.notna(row.get("content_type")) else "keyword article",
        "main_intent": str(row.get("main_intent", "informational")) if pd.notna(row.get("main_intent")) else "informational",
        "word_count": wc,
        "char_count": cc,
        "avg_position": clean_num(row.get("avg_position"), 0.0),
        "search_volume": int(clean_num(row.get("search_volume"), 0)),
        "competition": clean_num(row.get("competition"), 0.0),
        "cpc": clean_num(row.get("cpc"), 0.0),
        "content_age_days": int(clean_num(row.get("content_age_days"), 180)),
        "days_since_last_update": int(clean_num(row.get("days_since_last_update"), 30)),
        "sessions_90d": int(clean_num(row.get("sessions_90d"), 0)),
        "pageviews_90d": int(clean_num(row.get("pageviews_90d"), 0)),
        "users_90d": int(clean_num(row.get("users_90d"), 0)),
        "engaged_sessions_90d": int(clean_num(row.get("engaged_sessions_90d"), 0)),
        "days_with_sessions": int(clean_num(row.get("days_with_sessions"), 0)),
        "engagement_rate": clean_num(row.get("engagement_rate"), 0.0),
        "scroll_rate": clean_num(row.get("scroll_rate"), 0.0),
        "scroll_events_90d": int(clean_num(row.get("scroll_events_90d"), 0)),
        "ai_sessions_90d": int(clean_num(row.get("ai_sessions_90d"), 0)),
        "ai_traffic_pct": clean_num(row.get("ai_traffic_pct"), 0.0),
        "actual_impressions_90d": int(clean_num(row.get("impressions_90d"), 0)),
    }


# ==============================================================================
# CURATED REAL-WORLD ARTICLE SCENARIOS FOR LIVE SYSTEM TESTING (ALL 6 CONCLUSIONS)
# ==============================================================================
DEMO_PRESETS = {
    "Scenario 1: Striking-Distance Pillar Article (Rank #8.3 -> SERP Push)": {
        "description": (
            "In-depth 3,292-word informational guide ranking at #8.3 on Google for a 1,600/mo keyword "
            "with 222 organic sessions. System predicts ~28,239 impressions and recommends a Top-3 SERP push."
        ),
        "content_type": "keyword article",
        "main_intent": "informational",
        "word_count": 3292,
        "char_count": 20892,
        "avg_position": 8.3,
        "search_volume": 1600,
        "competition": 0.0,
        "cpc": 0.94,
        "content_age_days": 390,
        "days_since_last_update": 7,
        "sessions_90d": 222,
        "pageviews_90d": 232,
        "users_90d": 224,
        "engaged_sessions_90d": 0,
        "days_with_sessions": 60,
        "engagement_rate": 0.0,
        "scroll_rate": 4.74,
        "scroll_events_90d": 11,
        "ai_sessions_90d": 0,
        "ai_traffic_pct": 0.0,
        "actual_impressions_90d": 2129,
    },
    "Scenario 2: Thin Article on High-Demand Keyword (650 Words -> Content Expansion)": {
        "description": (
            "Short 650-word article targeting a high-demand keyword (2,400 monthly searches) at Rank #11.5 (Page 2). "
            "System predicts ~1,897 impressions and prescribes Content Expansion (>1,500 words)."
        ),
        "content_type": "keyword article",
        "main_intent": "informational",
        "word_count": 650,
        "char_count": 4030,
        "avg_position": 11.5,
        "search_volume": 2400,
        "competition": 0.0,
        "cpc": 0.94,
        "content_age_days": 390,
        "days_since_last_update": 145,
        "sessions_90d": 35,
        "pageviews_90d": 40,
        "users_90d": 34,
        "engaged_sessions_90d": 1,
        "days_with_sessions": 24,
        "engagement_rate": 2.86,
        "scroll_rate": 6.0,
        "scroll_events_90d": 2,
        "ai_sessions_90d": 0,
        "ai_traffic_pct": 0.0,
        "actual_impressions_90d": 45,
    },
    "Scenario 3: Stale Authority Guide (240 Days Since Update -> Freshness Refresh)": {
        "description": (
            "Comprehensive 3,800-word commercial guide at Rank #6.5 with 1,200 search volume, untouched for "
            "240 days (>180d threshold). System predicts ~5,977 impressions and prescribes a Freshness Refresh."
        ),
        "content_type": "keyword article",
        "main_intent": "commercial",
        "word_count": 3800,
        "char_count": 23560,
        "avg_position": 6.5,
        "search_volume": 1200,
        "competition": 0.0,
        "cpc": 0.94,
        "content_age_days": 390,
        "days_since_last_update": 240,
        "sessions_90d": 120,
        "pageviews_90d": 138,
        "users_90d": 115,
        "engaged_sessions_90d": 2,
        "days_with_sessions": 44,
        "engagement_rate": 1.67,
        "scroll_rate": 6.0,
        "scroll_events_90d": 8,
        "ai_sessions_90d": 0,
        "ai_traffic_pct": 0.0,
        "actual_impressions_90d": 950,
    },
    "Scenario 4: Under-Indexed Top-3 Comparison (Rank #2.5 -> Comprehensive SEO Refresh)": {
        "description": (
            "2,100-word commercial comparison article at Rank #2.5 with 1,800 search volume and 160 sessions, "
            "getting only 400 impressions. System predicts ~45,921 impressions and prescribes Comprehensive SEO Refresh."
        ),
        "content_type": "comparison article",
        "main_intent": "commercial",
        "word_count": 2100,
        "char_count": 13020,
        "avg_position": 2.5,
        "search_volume": 1800,
        "competition": 0.0,
        "cpc": 0.94,
        "content_age_days": 390,
        "days_since_last_update": 45,
        "sessions_90d": 160,
        "pageviews_90d": 184,
        "users_90d": 154,
        "engaged_sessions_90d": 3,
        "days_with_sessions": 51,
        "engagement_rate": 1.88,
        "scroll_rate": 6.0,
        "scroll_events_90d": 11,
        "ai_sessions_90d": 0,
        "ai_traffic_pct": 0.0,
        "actual_impressions_90d": 400,
    },
    "Scenario 5: Healthy Flagship Page at Capacity (Rank #2.0 -> Hold & Monitor)": {
        "description": (
            "High-performing 2,600-word flagship article at Rank #2.0 already earning 95,000 impressions "
            "(vs. ~108,037 predicted, gap +0.13 log < 0.50). System recommends Hold & Monitor to avoid disruptive rewrites."
        ),
        "content_type": "keyword article",
        "main_intent": "informational",
        "word_count": 2600,
        "char_count": 16120,
        "avg_position": 2.0,
        "search_volume": 1500,
        "competition": 0.0,
        "cpc": 0.94,
        "content_age_days": 390,
        "days_since_last_update": 20,
        "sessions_90d": 310,
        "pageviews_90d": 356,
        "users_90d": 298,
        "engaged_sessions_90d": 6,
        "days_with_sessions": 70,
        "engagement_rate": 1.94,
        "scroll_rate": 6.0,
        "scroll_events_90d": 21,
        "ai_sessions_90d": 0,
        "ai_traffic_pct": 0.0,
        "actual_impressions_90d": 95000,
    },
    "Scenario 6: Low-Potential 3,500-Word Zero-Demand Trap (Rank #85 -> Do Not Rewrite)": {
        "description": (
            "Long 3,500-word article buried at Rank #85 with zero search volume and 1 session. "
            "Naive Word Count heuristic falsely predicts ~474 impressions due to length, whereas our System "
            "accurately predicts 64 impressions and warns editors NOT to rewrite."
        ),
        "content_type": "keyword article",
        "main_intent": "informational",
        "word_count": 3500,
        "char_count": 21700,
        "avg_position": 85.0,
        "search_volume": 0,
        "competition": 0.0,
        "cpc": 0.94,
        "content_age_days": 390,
        "days_since_last_update": 40,
        "sessions_90d": 1,
        "pageviews_90d": 1,
        "users_90d": 1,
        "engaged_sessions_90d": 0,
        "days_with_sessions": 4,
        "engagement_rate": 0.0,
        "scroll_rate": 6.0,
        "scroll_events_90d": 0,
        "ai_sessions_90d": 0,
        "ai_traffic_pct": 0.0,
        "actual_impressions_90d": 45,
    },
    "Scenario 7: Page-2 Transactional Comparison (Rank #12.5 -> SERP Push)": {
        "description": (
            "1,950-word transactional comparison sitting on Page 2 (Rank #12.5) for a 2,900-volume keyword. "
            "System predicts ~5,206 impressions (+4,896 upside) and recommends a SERP Striking Distance Push."
        ),
        "content_type": "comparison article",
        "main_intent": "transactional",
        "word_count": 1950,
        "char_count": 12090,
        "avg_position": 12.5,
        "search_volume": 2900,
        "competition": 0.0,
        "cpc": 0.94,
        "content_age_days": 390,
        "days_since_last_update": 95,
        "sessions_90d": 85,
        "pageviews_90d": 98,
        "users_90d": 82,
        "engaged_sessions_90d": 2,
        "days_with_sessions": 37,
        "engagement_rate": 2.35,
        "scroll_rate": 6.0,
        "scroll_events_90d": 6,
        "ai_sessions_90d": 0,
        "ai_traffic_pct": 0.0,
        "actual_impressions_90d": 310,
    },
    "Scenario 8: Short High-Intent Landing Page (550 Words -> Content Expansion)": {
        "description": (
            "Thin 550-word transactional page on Page 1 (Rank #7.0) with 3,200 monthly search volume. "
            "System predicts ~10,937 impressions (+10,417 upside) and prescribes Content Expansion."
        ),
        "content_type": "keyword article",
        "main_intent": "transactional",
        "word_count": 550,
        "char_count": 3410,
        "avg_position": 7.0,
        "search_volume": 3200,
        "competition": 0.0,
        "cpc": 0.94,
        "content_age_days": 390,
        "days_since_last_update": 60,
        "sessions_90d": 90,
        "pageviews_90d": 103,
        "users_90d": 86,
        "engaged_sessions_90d": 2,
        "days_with_sessions": 38,
        "engagement_rate": 2.22,
        "scroll_rate": 6.0,
        "scroll_events_90d": 6,
        "ai_sessions_90d": 0,
        "ai_traffic_pct": 0.0,
        "actual_impressions_90d": 520,
    },
}


def classify_triage_reason(raw_inputs: dict, pred_log: float, actual_log: float):
    """Assign an automated operational reason code and editorial action playbook."""
    gap = pred_log - actual_log
    word_count = float(raw_inputs.get("word_count", 0))
    avg_pos = float(raw_inputs.get("avg_position", 0))
    days_upd = float(raw_inputs.get("days_since_last_update", 0))
    actual_impr = float(raw_inputs.get("actual_impressions_90d", 0))

    if gap < 0.50:
        if pred_log < 4.5:
            return (
                "low_potential_deprioritize",
                "DO NOT REWRITE (Low Potential / Low Demand)",
                "Predicted search visibility is low despite current attributes. Save editorial bandwidth for higher-ROI pages."
            )
        return (
            "performing_at_capacity",
            "HOLD & MONITOR (Performing at or above Expected Capacity)",
            "Realized impressions already match or exceed structural model expectations. Avoid disruptive rewrites."
        )

    if word_count < 1000 and gap >= 0.50:
        return (
            "thin_content_expansion",
            "CONTENT EXPANSION (High Demand, Thin Article)",
            f"Article is only {int(word_count)} words but targets strong demand/engagement signals. Expand to 1,500–2,500+ words with structured H2s and comparison tables."
        )
    if days_upd > 180 and actual_impr >= 50:
        return (
            "stale_authority_decay",
            f"FRESHNESS REFRESH ({int(days_upd)} Days Since Last Update)",
            "Strong authority article suffering from freshness decay. Update statistics, current-year examples, and republish timestamp."
        )
    if 3.0 < avg_pos <= 20.0:
        return (
            "striking_distance_opportunity",
            f"SERP STRIKING DISTANCE PUSH (Current Rank #{avg_pos:.1f} -> Target Top 3)",
            "Page sits in the rank #4–#20 striking zone with high latent visibility. Inject internal links from authority pages, add FAQ schema, and tighten title/H2 intent alignment."
        )
    return (
        "under_indexed_refresh",
        "COMPREHENSIVE SEO REFRESH (Under-Indexed High Potential)",
        "Page exhibits strong underlying engagement/demand signals relative to current impressions. Audit technical indexing, internal link equity, and SERP snippet CTR."
    )


def predict_from_raw_inputs(raw_inputs: dict, bundle: dict = None) -> dict:
    """Transform a single dictionary of real SEO values into 48 features and run all 5 models."""
    if bundle is None:
        bundle = train_and_save_bundle()

    enriched = derive_tiers(raw_inputs)
    row_df = pd.DataFrame([enriched])
    X_row = engineer_features_dataframe(row_df, reference_feature_cols=bundle["feature_cols"])

    interaction_signals = bundle["interaction_signals"]
    pred_dummy = float(bundle["y_train_mean"])
    pred_uni = float(bundle["lr_univariate"].predict(X_row[["log_word_count"]])[0])
    pred_ridge = float(bundle["poly_ridge_pipe"].predict(X_row[interaction_signals])[0])
    pred_hgbr = float(bundle["hgbr_model"].predict(X_row)[0])
    pred_ensemble = 0.50 * pred_ridge + 0.50 * pred_hgbr

    actual_impr = float(raw_inputs.get("actual_impressions_90d", 0.0))
    actual_log = float(np.log1p(max(0.0, actual_impr)))
    opportunity_gap = pred_ensemble - actual_log

    reason_code, badge_title, playbook_text = classify_triage_reason(raw_inputs, pred_ensemble, actual_log)

    models_out = [
        {
            "Model": "Baseline 1: Mean Dummy Regressor",
            "Pred Log(1+Impr)": round(pred_dummy, 4),
            "Pred 90d Impressions": int(round(np.expm1(max(0.0, pred_dummy)))),
        },
        {
            "Model": "Baseline 2: Univariate Word Count Rule",
            "Pred Log(1+Impr)": round(pred_uni, 4),
            "Pred 90d Impressions": int(round(np.expm1(max(0.0, pred_uni)))),
        },
        {
            "Model": f"Candidate A: Poly Interaction Ridge (α={bundle['best_alpha']:.1f})",
            "Pred Log(1+Impr)": round(pred_ridge, 4),
            "Pred 90d Impressions": int(round(np.expm1(max(0.0, pred_ridge)))),
        },
        {
            "Model": "Candidate B: HistGradientBoosting (HGBR)",
            "Pred Log(1+Impr)": round(pred_hgbr, 4),
            "Pred 90d Impressions": int(round(np.expm1(max(0.0, pred_hgbr)))),
        },
        {
            "Model": "CHAMPION: Stacking Ensemble (50% Ridge + 50% HGBR)",
            "Pred Log(1+Impr)": round(pred_ensemble, 4),
            "Pred 90d Impressions": int(round(np.expm1(max(0.0, pred_ensemble)))),
        },
    ]

    key_engineered = {
        "serp_rank_power (1 / pos)": round(float(X_row["serp_rank_power"].iloc[0]), 4),
        "demand_position_synergy": round(float(X_row["demand_position_synergy"].iloc[0]), 4),
        "engagement_volume_synergy": round(float(X_row["engagement_volume_synergy"].iloc[0]), 4),
        "depth_engagement_synergy": round(float(X_row["depth_engagement_synergy"].iloc[0]), 4),
        "engaged_session_ratio": round(float(X_row["engaged_session_ratio"].iloc[0]), 4),
        "log_word_count": round(float(X_row["log_word_count"].iloc[0]), 4),
        "log_search_volume": round(float(X_row["log_search_volume"].iloc[0]), 4),
        "log_sessions_90d": round(float(X_row["log_sessions_90d"].iloc[0]), 4),
    }

    return {
        "pred_log_ensemble": pred_ensemble,
        "pred_impressions_ensemble": int(round(np.expm1(max(0.0, pred_ensemble)))),
        "pred_log_ridge": pred_ridge,
        "pred_impressions_ridge": int(round(np.expm1(max(0.0, pred_ridge)))),
        "pred_log_hgbr": pred_hgbr,
        "pred_impressions_hgbr": int(round(np.expm1(max(0.0, pred_hgbr)))),
        "pred_log_univariate": pred_uni,
        "pred_impressions_univariate": int(round(np.expm1(max(0.0, pred_uni)))),
        "actual_impressions": int(round(actual_impr)),
        "actual_log": actual_log,
        "opportunity_gap_log": opportunity_gap,
        "impression_upside_delta": int(round(np.expm1(max(0.0, pred_ensemble)) - actual_impr)),
        "reason_code": reason_code,
        "badge_title": badge_title,
        "playbook_text": playbook_text,
        "models_comparison": models_out,
        "key_engineered_signals": key_engineered,
        "derived_tiers": {
            "position_tier": enriched["position_tier"],
            "freshness_tier": enriched["freshness_tier"],
            "competition_level": enriched["competition_level"],
            "age_tier_order": enriched["age_tier_order"],
        },
    }


def simulate_rank_curve(raw_inputs: dict, bundle: dict = None) -> pd.DataFrame:
    """Simulate how the Champion Ensemble vs. Word Count Baseline behaves as Google Rank moves from #1 to #30."""
    if bundle is None:
        bundle = train_and_save_bundle()

    positions = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0]
    rows = []
    for pos in positions:
        sim_input = dict(raw_inputs)
        sim_input["avg_position"] = pos
        sim_input.pop("position_tier", None)
        res = predict_from_raw_inputs(sim_input, bundle)
        rows.append({
            "Google Rank Position": f"Rank #{pos:.0f}",
            "avg_position": pos,
            "SERP Power (1/pos)": round(1.0 / pos, 3),
            "Champion Ensemble (Pred Impressions)": res["pred_impressions_ensemble"],
            "Champion Log(1+Impr)": round(res["pred_log_ensemble"], 3),
            "Baseline 2: Word Count Rule (Pred Impressions)": res["pred_impressions_univariate"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Building/Loading Champion Live Inference Bundle & Unseen Test Dataset CSV...")
    b = train_and_save_bundle(force_retrain=True)
    print(f"Bundle path: {get_bundle_path()}")
    print(f"Unseen Test Set CSV (6,163 rows across 7 unseen clients): {get_test_csv_path()}")
