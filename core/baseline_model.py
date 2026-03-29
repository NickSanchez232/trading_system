import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
from datetime import datetime


def load_and_prepare_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_dataset.csv')
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} rows")

    # Only train on strong signals
    df_train = df[df['label'].isin(['strong_positive', 'strong_negative'])].copy()
    print(f"Strong signals only: {len(df_train)} rows")

    # Convert label to numeric: 1 = strong_positive, 0 = strong_negative
    df_train['target'] = (df_train['label'] == 'strong_positive').astype(int)

    # Select feature columns (drop non-feature columns)
    drop_cols = ['symbol', 'date', 'return_pct', 'label', 'target',
                 'market_regime', 'fed_rate_direction', 'fear_greed_rating',
                 'credit_spread_direction', 'earnings_beat']

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
    # Earlier data for training, recent data for testing
    split_point = int(len(X) * 0.8)
    X_train = X.iloc[:split_point]
    y_train = y.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_test = y.iloc[split_point:]

    print(f"\nTraining set: {len(X_train)} rows")
    print(f"Test set: {len(X_test)} rows")

    # Create XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
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

    # Feature importance - which signals matter most
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 15 Most Important Features:")
    for _, row in importance.head(15).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return model, importance

def save_model(model, importance):
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f'xgboost_v1_{timestamp}.json')
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    importance_path = os.path.join(model_dir, f'feature_importance_{timestamp}.csv')
    importance.to_csv(importance_path, index=False)
    print(f"Feature importance saved to {importance_path}")


if __name__ == "__main__":
    X, y, feature_cols = load_and_prepare_data()
    model, importance = train_model(X, y, feature_cols)
    save_model(model, importance)