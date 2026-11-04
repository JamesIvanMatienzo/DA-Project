import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def resolve_data_path():
    """Resolve the path to the raw dataset from various working directories."""
    candidate_paths = [
        os.path.join('data', 'raw', 'content_refresh_anonymized.csv'),
        os.path.join('..', 'data', 'raw', 'content_refresh_anonymized.csv'),
        os.path.join('..', '..', 'data', 'raw', 'content_refresh_anonymized.csv')
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not locate 'content_refresh_anonymized.csv'. Ensure data/raw exists.")


def evaluate_metrics(y_true, y_pred, name="Model"):
    """Compute standard regression and monotonic ranking metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    try:
        rho, _ = spearmanr(y_true, y_pred)
        if np.isnan(rho):
            rho = 0.0
    except Exception:
        rho = 0.0
    return {
        'Model': name,
        'R2 Score': round(r2, 4),
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'Spearman rho': round(rho, 4)
    }


def main():
    print("=" * 70)
    print("FlyRank Ranking Signal Analysis — Champion Model Training Pipeline")
    print("=" * 70)

    # 1. Load Raw Dataset
    data_path = resolve_data_path()
    print(f"\n[1/6] Ingesting dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"      Rows: {len(df):,}, Columns: {df.shape[1]}")

    # 2. Target Formulation
    print("\n[2/6] Formulating Continuous Target: log(1 + impressions_90d)")
    df['target_log_impressions'] = np.log1p(df['impressions_90d'])

    # 3. Domain-Specific Feature Engineering
    print("\n[3/6] Engineering Domain Signals & Assembling Feature Vector (48 features)...")
    
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

    # Categorical One-Hot Dummies
    cat_cols = ['content_type', 'main_intent', 'competition_level', 'freshness_tier', 'position_tier']
    dummy_dfs = [pd.get_dummies(df[c].fillna('unknown'), prefix=c, drop_first=True, dtype=float) for c in cat_cols]

    meta_cols = ['content_id', 'client_id', 'target_log_impressions']
    df_features = pd.concat([df[meta_cols], df[numeric_cols]] + dummy_dfs, axis=1)
    df_features = df_features.loc[:, ~df_features.columns.duplicated()]

    feature_cols = [c for c in df_features.columns if c not in meta_cols]
    print(f"      Total Clean Engineered Features: {len(feature_cols)}")

    # 4. Grouped Split (Zero-Leakage Multi-Tenant Partition)
    print("\n[4/6] Executing Grouped Holdout Split by client_id (80/20 train/test)...")
    X = df_features[feature_cols]
    y = df_features['target_log_impressions']
    groups = df_features['client_id']

    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    clients_train, clients_test = groups.iloc[train_idx], groups.iloc[test_idx]

    # Assert Disjoint Sets
    assert set(clients_train).isdisjoint(set(clients_test)), "CRITICAL: Client leakage detected!"
    print(f"      Train Set: {len(X_train):,} items ({clients_train.nunique()} unique clients)")
    print(f"      Test Set:  {len(X_test):,} items ({clients_test.nunique()} unique unseen clients)")

    # 5. Fit Baselines and Candidates
    print("\n[5/6] Training Baseline Benchmarks and Candidate Estimators...")
    results = []

    # Baseline 1: Mean Dummy
    y_pred_dummy = np.full_like(y_test, y_train.mean())
    results.append(evaluate_metrics(y_test, y_pred_dummy, "Baseline 1: Mean Dummy Regressor"))

    # Baseline 2: Univariate Word Count OLS
    lr_univariate = LinearRegression().fit(X_train[['log_word_count']], y_train)
    y_pred_univariate = lr_univariate.predict(X_test[['log_word_count']])
    results.append(evaluate_metrics(y_test, y_pred_univariate, "Baseline 2: Univariate Rule (log_word_count)"))

    # Candidate A: Polynomial Interaction Ridge
    print("      Fitting Candidate A: Polynomial Interaction Ridge (RidgeCV)...")
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
    best_alpha = poly_ridge_pipe.named_steps['regressor'].alpha_
    y_pred_ridge = poly_ridge_pipe.predict(X_test[interaction_signals])
    results.append(evaluate_metrics(y_test, y_pred_ridge, f"Candidate A: Poly Ridge (alpha={best_alpha:.2f})"))

    # Candidate B: HistGradientBoostingRegressor
    print("      Fitting Candidate B: HistGradientBoostingRegressor...")
    hgbr_model = HistGradientBoostingRegressor(
        max_iter=350,
        learning_rate=0.03,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=2.0,
        random_state=42
    )
    hgbr_model.fit(X_train, y_train)
    y_pred_hgbr = hgbr_model.predict(X_test)
    results.append(evaluate_metrics(y_test, y_pred_hgbr, "Candidate B: HistGradientBoosting"))

    # Champion: Blended Stacking Ensemble
    print("      Evaluating Champion Stacking Ensemble (0.50 * Ridge + 0.50 * HGBR)...")
    y_pred_ensemble = 0.50 * y_pred_ridge + 0.50 * y_pred_hgbr
    results.append(evaluate_metrics(y_test, y_pred_ensemble, "Champion: Stacking Ensemble (Ridge + HGBR)"))

    # 6. Display Benchmark Performance Table
    print("\n[6/6] Official Benchmark Performance Table on Held-Out Unseen Clients:")
    results_df = pd.DataFrame(results)
    print("-" * 75)
    print(results_df.to_string(index=False))
    print("-" * 75)
    print("\nTraining and validation complete.")


if __name__ == '__main__':
    main()
