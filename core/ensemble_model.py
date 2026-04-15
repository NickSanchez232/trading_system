import os
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib


# XGBoost best parameters
XGB_PARAMS = {
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

# LSTM best parameters
LSTM_PARAMS = {
    "_2w": {
        'sequence_length': 3,
        'hidden_size': 128,
        'num_layers': 1,
        'dropout': 0.43,
    },
    "_30d": {
        'sequence_length': 3,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.44,
    },
}


# Import LSTM architecture from lstm_model
from lstm_model import StockLSTM, Attention


def load_data(timeframe="_2w"):
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_dataset.csv')
    df = pd.read_csv(data_path)

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

    df_train = df_train.sort_values(['symbol', 'date']).reset_index(drop=True)

    X_raw = df_train[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y_raw = df_train['target'].values
    symbols = df_train['symbol'].values
    dates = df_train['date'].values

    # Valid mask for XGBoost (same as baseline_model.py)
    valid_mask = pd.DataFrame(X_raw).notna().sum(axis=1) >= (len(feature_cols) * 0.5)

    return X_raw, y_raw, symbols, dates, feature_cols, valid_mask, df_train


def train_xgboost(X_raw, y_raw, valid_mask, feature_cols, timeframe="_2w"):
    print("\n--- Training XGBoost ---")

    X = pd.DataFrame(X_raw, columns=feature_cols)[valid_mask]
    X = X.apply(pd.to_numeric, errors='coerce')
    y = y_raw[valid_mask]

    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    neg_count = len(y_train[y_train == 0])
    pos_count = len(y_train[y_train == 1])
    scale_weight = neg_count / pos_count

    params = XGB_PARAMS.get(timeframe, XGB_PARAMS["_2w"])

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

    xgb_probs = model.predict_proba(X_test)[:, 1]
    xgb_preds = model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    print(f"  XGBoost Accuracy: {xgb_acc:.2%}")

    return model, xgb_probs, y_test, X_test.index


def train_lstm_model(X_raw, y_raw, symbols, feature_cols, timeframe="_2w"):
    print("\n--- Training LSTM ---")

    params = LSTM_PARAMS.get(timeframe, LSTM_PARAMS["_2w"])
    seq_len = params['sequence_length']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(X_raw, 0))

    X_sequences = []
    y_sequences = []
    seq_indices = []

    unique_symbols = np.unique(symbols)
    for symbol in unique_symbols:
        mask = symbols == symbol
        indices = np.where(mask)[0]
        X_symbol = X_scaled[mask]
        y_symbol = y_raw[mask]

        for i in range(seq_len, len(X_symbol)):
            X_sequences.append(X_symbol[i - seq_len:i])
            y_sequences.append(y_symbol[i])
            seq_indices.append(indices[i])

    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)
    seq_indices = np.array(seq_indices)

    split_point = int(len(X_seq) * 0.8)
    X_train = X_seq[:split_point]
    y_train = y_seq[:split_point]
    X_test = X_seq[split_point:]
    y_test = y_seq[split_point:]
    test_indices = seq_indices[split_point:]

    input_size = len(feature_cols)
    model = StockLSTM(
        input_size=input_size,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        bidirectional=True,
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)

    from torch.utils.data import Dataset, DataLoader

    class SeqDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=256, shuffle=True, drop_last=True)
    test_loader = DataLoader(SeqDataset(X_test, y_test), batch_size=256, shuffle=False, drop_last=True)

    best_acc = 0
    best_state = None

    for epoch in range(50):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx).squeeze(-1)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    out = model(bx).squeeze(-1)
                    preds.extend((out >= 0.5).float().numpy())
                    trues.extend(by.numpy())
            acc = accuracy_score(trues, preds)
            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict().copy()
            print(f"  Epoch {epoch+1}/50 | Accuracy: {acc:.2%}")

    model.load_state_dict(best_state)
    model.eval()

    all_probs = []
    all_true = []
    with torch.no_grad():
        for bx, by in test_loader:
            out = model(bx).squeeze(-1)
            all_probs.extend(out.numpy())
            all_true.extend(by.numpy())

    lstm_acc = accuracy_score(all_true, (np.array(all_probs) >= 0.5).astype(int))
    print(f"  LSTM Accuracy: {lstm_acc:.2%}")

    return model, np.array(all_probs), np.array(all_true), test_indices


def build_ensemble(xgb_probs, xgb_y, lstm_probs, lstm_y):
    print("\n--- Building Ensemble ---")

    # Match sizes (LSTM might have fewer due to drop_last)
    min_len = min(len(xgb_probs), len(lstm_probs))
    xgb_p = xgb_probs[-min_len:]
    lstm_p = lstm_probs[-min_len:]
    y_true = xgb_y[-min_len:]

    # Test multiple ensemble strategies
    print("\n  Strategy 1: Weighted Average (0.7 XGB + 0.3 LSTM)")
    combo_probs = 0.7 * xgb_p + 0.3 * lstm_p
    combo_preds = (combo_probs >= 0.5).astype(int)
    acc1 = accuracy_score(y_true, combo_preds)
    print(f"  Accuracy: {acc1:.2%}")
    print(classification_report(y_true, combo_preds, target_names=['Avoid', 'Buy']))

    print("  Strategy 2: Veto Rule (both must agree)")
    xgb_preds = (xgb_p >= 0.5).astype(int)
    lstm_preds = (lstm_p >= 0.5).astype(int)
    agree_mask = xgb_preds == lstm_preds
    if agree_mask.sum() > 0:
        veto_acc = accuracy_score(y_true[agree_mask], xgb_preds[agree_mask])
        agree_pct = agree_mask.sum() / len(agree_mask) * 100
        print(f"  Accuracy (when both agree): {veto_acc:.2%}")
        print(f"  Agreement rate: {agree_pct:.1f}% of signals")
        print(classification_report(y_true[agree_mask], xgb_preds[agree_mask], target_names=['Avoid', 'Buy']))

    print("  Strategy 3: Confidence Threshold (combined > 0.6)")
    high_conf_mask = combo_probs >= 0.6
    low_conf_mask = combo_probs < 0.4
    signal_mask = high_conf_mask | low_conf_mask
    if signal_mask.sum() > 0:
        conf_preds = (combo_probs[signal_mask] >= 0.5).astype(int)
        conf_acc = accuracy_score(y_true[signal_mask], conf_preds)
        signal_pct = signal_mask.sum() / len(signal_mask) * 100
        print(f"  Accuracy (high confidence only): {conf_acc:.2%}")
        print(f"  Signal rate: {signal_pct:.1f}% of opportunities")
        print(classification_report(y_true[signal_mask], conf_preds, target_names=['Avoid', 'Buy']))

    # Find optimal weight
    print("\n  Strategy 4: Optimal Weight Search")
    best_weight = 0.5
    best_acc = 0
    for w in np.arange(0.5, 0.95, 0.05):
        combo = w * xgb_p + (1 - w) * lstm_p
        preds = (combo >= 0.5).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_weight = w
    print(f"  Best weight: {best_weight:.2f} XGB / {1-best_weight:.2f} LSTM")
    print(f"  Best accuracy: {best_acc:.2%}")

    return best_weight, best_acc


if __name__ == "__main__":
    for tf in ["_2w", "_30d"]:
        print(f"\n{'='*60}")
        print(f"ENSEMBLE MODEL {tf}")
        print(f"{'='*60}")

        X_raw, y_raw, symbols, dates, feature_cols, valid_mask, df_train = load_data(tf)

        xgb_model, xgb_probs, xgb_y, xgb_idx = train_xgboost(X_raw, y_raw, valid_mask, feature_cols, tf)
        lstm_model, lstm_probs, lstm_y, lstm_idx = train_lstm_model(X_raw, y_raw, symbols, feature_cols, tf)

        best_weight, best_acc = build_ensemble(xgb_probs, xgb_y, lstm_probs, lstm_y)

        print(f"\n{'='*60}")
        print(f"ENSEMBLE RESULTS {tf}")
        print(f"{'='*60}")
        print(f"  XGBoost alone: {accuracy_score(xgb_y, (xgb_probs >= 0.5).astype(int)):.2%}")
        print(f"  LSTM alone: {accuracy_score(lstm_y, (np.array(lstm_probs) >= 0.5).astype(int)):.2%}")
        print(f"  Best ensemble: {best_acc:.2%} (weight: {best_weight:.2f}/{1-best_weight:.2f})")