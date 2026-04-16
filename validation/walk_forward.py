import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

BEST_PARAMS = {
    "_2w": {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'min_child_weight': 1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
    },
    "_30d": {
        'n_estimators': 1000,
        'max_depth': 8,
        'learning_rate': 0.03,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
    },
}


def load_data(timeframe="_2w"):
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_dataset.csv')
    df = pd.read_csv(data_path, low_memory=False)

    df = df[df['timeframe'] == timeframe]
    df_train = df[df['label'].isin(['strong_positive', 'strong_negative'])].copy()
    df_train['target'] = (df_train['label'] == 'strong_positive').astype(int)

    drop_cols = ['symbol', 'date', 'return_pct', 'label', 'target', 'timeframe',
                 'max_gain', 'max_drawdown',
                 'market_regime', 'fed_rate_direction', 'fear_greed_rating',
                 'credit_spread_direction', 'earnings_beat',
                 'revenue', 'revenue_growth', 'earnings_per_share', 'eps_growth',
                 'profit_margin', 'pe_ratio', 'sector_avg_pe', 'pe_vs_sector',
                 'free_cash_flow', 'fcf_yield', 'debt_to_equity',
                 'days_to_next_earnings',
                 'sector_momentum_vs_sp500', 'short_pct_of_float', 'short_ratio',
                 'analyst_total_buy', 'analyst_total_sell', 'analyst_total_hold',
                 'analyst_buy_sell_ratio', 'news_sentiment_score', 'article_count',
                 'fear_greed_score', 'insider_ownership_pct', 'unemployment_rate',
                 'fund_names', 'has_active_buyback']

    feature_cols = [c for c in df_train.columns if c not in drop_cols]

    # Sort by DATE for truly chronological walk-forward splits
    df_train = df_train.sort_values('date').reset_index(drop=True)

    X = df_train[feature_cols].apply(pd.to_numeric, errors='coerce')
    y = df_train['target']
    dates = df_train['date']

    valid_mask = X.notna().sum(axis=1) >= (len(feature_cols) * 0.5)
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    dates = dates[valid_mask].reset_index(drop=True)

    return X, y, dates, feature_cols


def walk_forward_validate(timeframe="_2w", n_splits=5):
    print(f"\nWalk-Forward Validation: {timeframe}")
    print("=" * 60)

    X, y, dates, feature_cols = load_data(timeframe)
    total_rows = len(X)

    print(f"Total rows: {total_rows}")
    print(f"Date range: {dates.min()} to {dates.max()}")

    params = BEST_PARAMS.get(timeframe, BEST_PARAMS["_2w"])

    # Split data into n_splits equal chunks by time
    chunk_size = total_rows // (n_splits + 1)

    all_accuracies = []
    all_buy_precisions = []
    all_avoid_precisions = []

    for i in range(n_splits):
        # Training data: all data up to this point
        train_end = chunk_size * (i + 1)
        test_start = train_end
        test_end = test_start + chunk_size

        if test_end > total_rows:
            test_end = total_rows

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[test_start:test_end]
        y_test = y.iloc[test_start:test_end]

        train_dates = dates.iloc[:train_end]
        test_dates = dates.iloc[test_start:test_end]

        if len(X_test) == 0:
            continue

        neg_count = len(y_train[y_train == 0])
        pos_count = len(y_train[y_train == 1])
        scale_weight = neg_count / pos_count

        model = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            min_child_weight=params['min_child_weight'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_weight,
        )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Avoid', 'Buy'], output_dict=True)

        buy_precision = report['Buy']['precision']
        avoid_precision = report['Avoid']['precision']

        all_accuracies.append(accuracy)
        all_buy_precisions.append(buy_precision)
        all_avoid_precisions.append(avoid_precision)

        print(f"\nSplit {i+1}/{n_splits}")
        print(f"  Train: {train_dates.min()} to {train_dates.max()} ({len(X_train)} rows)")
        print(f"  Test:  {test_dates.min()} to {test_dates.max()} ({len(X_test)} rows)")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Buy precision: {buy_precision:.2%}")
        print(f"  Avoid precision: {avoid_precision:.2%}")

    print(f"\n{'=' * 60}")
    print(f"WALK-FORWARD RESULTS: {timeframe}")
    print(f"{'=' * 60}")
    print(f"  Average accuracy: {np.mean(all_accuracies):.2%}")
    print(f"  Min accuracy:     {np.min(all_accuracies):.2%}")
    print(f"  Max accuracy:     {np.max(all_accuracies):.2%}")
    print(f"  Std deviation:    {np.std(all_accuracies):.2%}")
    print(f"  Avg buy precision:   {np.mean(all_buy_precisions):.2%}")
    print(f"  Avg avoid precision: {np.mean(all_avoid_precisions):.2%}")

    return all_accuracies


if __name__ == "__main__":
    for tf in ["_2w", "_30d"]:
        accuracies = walk_forward_validate(tf, n_splits=5)