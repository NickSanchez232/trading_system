import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import joblib
from datetime import datetime


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
                 'market_regime', 'fed_rate_direction', 'fear_greed_rating',
                 'credit_spread_direction', 'earnings_beat',
                 'revenue', 'revenue_growth', 'earnings_per_share', 'eps_growth',
                 'profit_margin', 'pe_ratio', 'sector_avg_pe', 'pe_vs_sector',
                 'free_cash_flow', 'fcf_yield', 'debt_to_equity',
                 'days_to_next_earnings',
                 'sector_momentum_vs_sp500', 'short_pct_of_float', 'short_ratio',
                 'analyst_total_buy', 'analyst_total_sell', 'analyst_total_hold',
                 'analyst_buy_sell_ratio', 'news_sentiment_score', 'article_count',
                 'fear_greed_score', 'insider_ownership_pct', 'unemployment_rate']

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


def train_model(X, y, feature_cols):
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

    # Create XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
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

    print("\nTop 15 Most Important Features:")
    for _, row in importance.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return model, importance

def tune_model(X, y, feature_cols):
    split_point = int(len(X) * 0.8)
    X_train = X.iloc[:split_point]
    y_train = y.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_test = y.iloc[split_point:]

    neg_count = len(y_train[y_train == 0])
    pos_count = len(y_train[y_train == 1])
    scale_weight = neg_count / pos_count

    print("\nTuning hyperparameters...")

    param_grid = {
        'n_estimators': [200, 300, 500, 800, 1000],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
    }

    base_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=scale_weight,
    )

    grid = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.2%}")

    # Test best model
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Avoid', 'Buy']))

    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Most Important Features:")
    for _, row in importance.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return best_model, importance


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
        print(f"TUNING {tf} MODEL")
        print(f"{'='*50}")
        X, y, feature_cols = load_and_prepare_data(tf)
        model, importance = tune_model(X, y, feature_cols)
        save_model(model, importance, tf)