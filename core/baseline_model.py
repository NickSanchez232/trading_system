import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
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


def load_and_prepare_data(timeframe="_2w"):
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_dataset.csv')
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} rows")

    # Filter by timeframe
    df = df[df['timeframe'] == timeframe]
    print(f"Filtered to {timeframe}: {len(df)} rows")

    # Only train on strong signals
    df_train = df[df['label'].isin(['strong_positive', 'strong_negative'])].copy()
    print(f"Strong signals only: {len(df_train)} rows")

    # Convert label to numeric: 1 = strong_positive, 0 = strong_negative
    df_train['target'] = (df_train['label'] == 'strong_positive').astype(int)

    # Drop non-feature columns and snapshot features that leak future data
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

    print(f"Features: {len(feature_cols)}")
    print(f"Feature list: {feature_cols}")

    X = df_train[feature_cols]
    y = df_train['target']

    # Drop rows with too many NaN values
    valid_mask = X.notna().sum(axis=1) >= (len(feature_cols) * 0.5)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"Rows with sufficient data: {len(X)}")
    print(f"  Positive (buy): {y.sum()}")
    print(f"  Negative (avoid): {len(y) - y.sum()}")

    return X, y, feature_cols


def train_model(X, y, feature_cols, timeframe="_2w"):
    # Split by time - not randomly
    split_point = int(len(X) * 0.8)
    X_train = X.iloc[:split_point]
    y_train = y.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_test = y.iloc[split_point:]

    print(f"\nTraining set: {len(X_train)} rows")
    print(f"Test set: {len(X_test)} rows")

    # Calculate class weight to fix imbalance
    neg_count = len(y_train[y_train == 0])
    pos_count = len(y_train[y_train == 1])
    scale_weight = neg_count / pos_count

    # Get best parameters for this timeframe
    params = BEST_PARAMS.get(timeframe, BEST_PARAMS["_2w"])

    # Create XGBoost model with tuned parameters
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

    # Train the model
    print("\nTraining XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2%}")

    # Detailed report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Avoid', 'Buy']))

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nAll feature importanceb:")
    for _, row in importance.head(42).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return model, importance


def save_model(model, importance, timeframe=""):
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f'xgboost{timeframe}_{timestamp}.json')
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    importance_path = os.path.join(model_dir, f'feature_importance{timeframe}_{timestamp}.csv')
    importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")


if __name__ == "__main__":
    for tf in ["_2w", "_30d"]:
        print(f"\n{'='*50}")
        print(f"TRAINING {tf} MODEL")
        print(f"{'='*50}")
        X, y, feature_cols = load_and_prepare_data(tf)
        model, importance = train_model(X, y, feature_cols, tf)
        save_model(model, importance, tf)