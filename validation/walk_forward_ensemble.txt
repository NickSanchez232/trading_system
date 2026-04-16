import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Add parent and core directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from core.lstm_model import StockLSTM, Attention


# ============================================================
# PARAMETERS (same as your tuned models)
# ============================================================

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

LSTM_PARAMS = {
    "_2w": {
        'sequence_length': 3,
        'hidden_size': 128,
        'num_layers': 1,
        'dropout': 0.43,
        'learning_rate': 0.0044,
        'batch_size': 256,
        'weight_decay': 0.0000012,
    },
    "_30d": {
        'sequence_length': 3,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.44,
        'learning_rate': 0.0074,
        'batch_size': 64,
        'weight_decay': 0.000188,
    },
}

# Confidence thresholds to test
CONFIDENCE_THRESHOLDS = [0.55, 0.60, 0.65, 0.70]

# XGB/LSTM weight ratios to test
WEIGHT_RATIOS = [
    (0.80, 0.20),
    (0.85, 0.15),
    (0.90, 0.10),
    (0.95, 0.05),
]

# Drop columns — exact same as your baseline_model.py and ensemble_model.py
DROP_COLS = [
    'symbol', 'date', 'return_pct', 'label', 'target', 'timeframe',
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
    'fund_names', 'has_active_buyback',
]


# ============================================================
# DATA LOADING
# ============================================================

def load_data(timeframe="_2w"):
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_dataset.csv')
    df = pd.read_csv(data_path, low_memory=False)

    df = df[df['timeframe'] == timeframe]
    df_train = df[df['label'].isin(['strong_positive', 'strong_negative'])].copy()
    df_train['target'] = (df_train['label'] == 'strong_positive').astype(int)

    feature_cols = [c for c in df_train.columns if c not in DROP_COLS]

    # Sort by DATE for walk-forward splitting (chronological order)
    df_train = df_train.sort_values('date').reset_index(drop=True)

    print(f"Loaded {len(df_train)} rows for {timeframe}")
    print(f"Features: {len(feature_cols)}")
    print(f"Date range: {df_train['date'].iloc[0]} to {df_train['date'].iloc[-1]}")
    print(f"Positive: {df_train['target'].sum()} | Negative: {len(df_train) - df_train['target'].sum()}")

    return df_train, feature_cols


# ============================================================
# LSTM HELPER
# ============================================================

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_lstm_sequences(X_scaled, y_raw, symbols, seq_len):
    """Build sequences grouped by symbol for LSTM."""
    X_sequences = []
    y_sequences = []
    seq_indices = []  # track original row index

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

    return np.array(X_sequences), np.array(y_sequences), np.array(seq_indices)


def train_lstm_on_split(X_train_seq, y_train_seq, X_test_seq, y_test_seq,
                        feature_cols, timeframe, epochs=50):
    """Train LSTM and return test probabilities."""
    params = LSTM_PARAMS.get(timeframe, LSTM_PARAMS["_2w"])
    input_size = len(feature_cols)

    model = StockLSTM(
        input_size=input_size,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        bidirectional=True,
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'],
    )

    train_loader = DataLoader(
        SeqDataset(X_train_seq, y_train_seq),
        batch_size=params['batch_size'],
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        SeqDataset(X_test_seq, y_test_seq),
        batch_size=params['batch_size'],
        shuffle=False,
        drop_last=True,
    )

    if len(train_loader) == 0 or len(test_loader) == 0:
        return None, None

    best_acc = 0
    best_state = None

    for epoch in range(epochs):
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

    if best_state is None:
        return None, None

    model.load_state_dict(best_state)
    model.eval()

    all_probs = []
    all_true = []
    with torch.no_grad():
        for bx, by in test_loader:
            out = model(bx).squeeze(-1)
            all_probs.extend(out.numpy())
            all_true.extend(by.numpy())

    return np.array(all_probs), np.array(all_true)


# ============================================================
# WALK-FORWARD ENSEMBLE VALIDATION
# ============================================================

def walk_forward_ensemble(timeframe="_2w", n_splits=5):
    print(f"\n{'='*70}")
    print(f"ENSEMBLE WALK-FORWARD VALIDATION: {timeframe}")
    print(f"{'='*70}")

    df_train, feature_cols = load_data(timeframe)
    df_full = df_train.copy()  # Keep full dataframe for LSTM per-symbol splitting
    total_rows = len(df_train)

    X_raw = df_train[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y_raw = df_train['target'].values
    symbols = df_train['symbol'].values
    dates = df_train['date'].values

    # Valid mask for XGBoost — use .values to get numpy array for clean slicing
    valid_mask = (pd.DataFrame(X_raw).notna().sum(axis=1) >= (len(feature_cols) * 0.5)).values

    # LSTM params
    lstm_params = LSTM_PARAMS.get(timeframe, LSTM_PARAMS["_2w"])
    seq_len = lstm_params['sequence_length']

    # Split data into chunks for walk-forward
    chunk_size = total_rows // (n_splits + 1)

    # Results storage
    # Key: (threshold, xgb_weight) -> list of per-split results
    results = {}
    for thresh in CONFIDENCE_THRESHOLDS:
        for xgb_w, lstm_w in WEIGHT_RATIOS:
            results[(thresh, xgb_w)] = []

    xgb_only_results = []

    for split in range(n_splits):
        print(f"\n{'─'*70}")
        print(f"SPLIT {split+1}/{n_splits}")
        print(f"{'─'*70}")

        # Training data: all data up to this point
        train_end = chunk_size * (split + 1)
        test_start = train_end
        test_end = min(test_start + chunk_size, total_rows)

        if test_end <= test_start:
            continue

        train_dates = dates[:train_end]
        test_dates = dates[test_start:test_end]
        print(f"  Train: {train_dates[0]} to {train_dates[-1]} ({train_end} rows)")
        print(f"  Test:  {test_dates[0]} to {test_dates[-1]} ({test_end - test_start} rows)")

        # ── XGBoost ──────────────────────────────────────
        print(f"  Training XGBoost...")

        # Apply valid mask within train/test ranges (numpy boolean arrays)
        train_valid = valid_mask[:train_end]
        test_valid = valid_mask[test_start:test_end]

        X_train_raw = X_raw[:train_end][train_valid]
        X_xgb_train = pd.DataFrame(X_train_raw, columns=feature_cols)
        X_xgb_train = X_xgb_train.apply(pd.to_numeric, errors='coerce')
        y_xgb_train = y_raw[:train_end][train_valid]

        X_test_raw = X_raw[test_start:test_end][test_valid]
        X_xgb_test = pd.DataFrame(X_test_raw, columns=feature_cols)
        X_xgb_test = X_xgb_test.apply(pd.to_numeric, errors='coerce')
        y_xgb_test = y_raw[test_start:test_end][test_valid]

        if len(X_xgb_test) == 0:
            print(f"  No valid test rows, skipping split")
            continue

        neg_count = len(y_xgb_train[y_xgb_train == 0])
        pos_count = len(y_xgb_train[y_xgb_train == 1])
        scale_weight = neg_count / pos_count if pos_count > 0 else 1.0

        params = XGB_PARAMS.get(timeframe, XGB_PARAMS["_2w"])

        xgb_model = xgb.XGBClassifier(
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

        xgb_model.fit(X_xgb_train, y_xgb_train,
                       eval_set=[(X_xgb_test, y_xgb_test)], verbose=False)

        xgb_probs = xgb_model.predict_proba(X_xgb_test)[:, 1]
        xgb_preds = (xgb_probs >= 0.5).astype(int)
        xgb_acc = accuracy_score(y_xgb_test, xgb_preds)
        print(f"  XGBoost accuracy: {xgb_acc:.2%}")

        xgb_only_results.append({
            'split': split + 1,
            'accuracy': xgb_acc,
            'test_size': len(y_xgb_test),
        })

        # ── LSTM ─────────────────────────────────────────
        print(f"  Training LSTM...")

        # For LSTM, we need sequences grouped by symbol
        # Split the data by time, then build sequences per symbol within each split

        # Get train and test dataframes (before feature extraction)
        df_train_split = df_full.iloc[:train_end].copy()
        df_test_split = df_full.iloc[test_start:test_end].copy()

        # Sort each split by symbol+date for sequence building
        df_train_split = df_train_split.sort_values(['symbol', 'date']).reset_index(drop=True)
        df_test_split = df_test_split.sort_values(['symbol', 'date']).reset_index(drop=True)

        # Extract features and scale using training data only
        X_train_lstm_raw = df_train_split[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        y_train_lstm_raw = df_train_split['target'].values
        symbols_train_lstm = df_train_split['symbol'].values

        X_test_lstm_raw = df_test_split[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
        y_test_lstm_raw = df_test_split['target'].values
        symbols_test_lstm = df_test_split['symbol'].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_lstm_raw)
        X_test_scaled = scaler.transform(X_test_lstm_raw)

        # Build sequences from training data
        train_seqs, train_seq_y, _ = build_lstm_sequences(
            X_train_scaled, y_train_lstm_raw, symbols_train_lstm, seq_len
        )

        # For test sequences, we need context from training data per symbol
        # Combine last N training rows per symbol with test rows, then extract test-only sequences
        test_seqs_list = []
        test_y_list = []

        for symbol in np.unique(symbols_test_lstm):
            # Get this symbol's training data (last seq_len rows for context)
            sym_train_mask = symbols_train_lstm == symbol
            sym_train_X = X_train_scaled[sym_train_mask]
            sym_train_y = y_train_lstm_raw[sym_train_mask]

            # Get this symbol's test data
            sym_test_mask = symbols_test_lstm == symbol
            sym_test_X = X_test_scaled[sym_test_mask]
            sym_test_y = y_test_lstm_raw[sym_test_mask]

            if len(sym_test_X) == 0:
                continue

            # Combine context (last seq_len from train) + test
            if len(sym_train_X) >= seq_len:
                context_X = sym_train_X[-seq_len:]
                context_y = sym_train_y[-seq_len:]
                combined_X = np.vstack([context_X, sym_test_X])
                combined_y = np.concatenate([context_y, sym_test_y])
            else:
                combined_X = sym_test_X
                combined_y = sym_test_y

            # Build sequences from combined data
            offset = seq_len if len(sym_train_X) >= seq_len else 0
            for i in range(seq_len, len(combined_X)):
                # Only keep sequences where the target row is from test data
                if i >= offset:
                    test_seqs_list.append(combined_X[i - seq_len:i])
                    test_y_list.append(combined_y[i])

        if len(test_seqs_list) > 0:
            test_seqs = np.array(test_seqs_list)
            test_seq_y = np.array(test_y_list)
        else:
            test_seqs = np.array([])
            test_seq_y = np.array([])

        if len(train_seqs) == 0 or len(test_seqs) == 0:
            print(f"  Not enough sequences for LSTM, using XGBoost only")
            # Still record results with XGBoost only
            for thresh in CONFIDENCE_THRESHOLDS:
                for xgb_w, lstm_w in WEIGHT_RATIOS:
                    # Without LSTM, just use XGBoost with threshold
                    high_conf = (xgb_probs >= thresh) | (xgb_probs <= (1.0 - thresh))
                    if high_conf.sum() > 0:
                        conf_preds = (xgb_probs[high_conf] >= 0.5).astype(int)
                        conf_acc = accuracy_score(y_xgb_test[high_conf], conf_preds)
                        signal_rate = high_conf.sum() / len(high_conf) * 100
                    else:
                        conf_acc = xgb_acc
                        signal_rate = 100.0

                    results[(thresh, xgb_w)].append({
                        'split': split + 1,
                        'accuracy': conf_acc,
                        'signal_rate': signal_rate,
                        'signals': int(high_conf.sum()) if high_conf.sum() > 0 else len(xgb_probs),
                        'test_size': len(y_xgb_test),
                        'lstm_available': False,
                    })
            continue

        # Train LSTM
        lstm_probs, lstm_true = train_lstm_on_split(
            train_seqs, train_seq_y, test_seqs, test_seq_y,
            feature_cols, timeframe, epochs=50,
        )

        if lstm_probs is None:
            print(f"  LSTM training failed, using XGBoost only")
            for thresh in CONFIDENCE_THRESHOLDS:
                for xgb_w, lstm_w in WEIGHT_RATIOS:
                    high_conf = (xgb_probs >= thresh) | (xgb_probs <= (1.0 - thresh))
                    if high_conf.sum() > 0:
                        conf_preds = (xgb_probs[high_conf] >= 0.5).astype(int)
                        conf_acc = accuracy_score(y_xgb_test[high_conf], conf_preds)
                        signal_rate = high_conf.sum() / len(high_conf) * 100
                    else:
                        conf_acc = xgb_acc
                        signal_rate = 100.0
                    results[(thresh, xgb_w)].append({
                        'split': split + 1,
                        'accuracy': conf_acc,
                        'signal_rate': signal_rate,
                        'signals': int(high_conf.sum()) if high_conf.sum() > 0 else len(xgb_probs),
                        'test_size': len(y_xgb_test),
                        'lstm_available': False,
                    })
            continue

        lstm_acc = accuracy_score(lstm_true, (lstm_probs >= 0.5).astype(int))
        print(f"  LSTM accuracy: {lstm_acc:.2%}")

        # ── ENSEMBLE COMBINATIONS ────────────────────────
        # Match sizes — LSTM may have fewer samples due to drop_last
        min_len = min(len(xgb_probs), len(lstm_probs))
        xgb_p = xgb_probs[-min_len:]
        lstm_p = lstm_probs[-min_len:]
        y_matched = y_xgb_test[-min_len:]

        print(f"  Matched samples for ensemble: {min_len}")

        # Test all threshold + weight combinations
        for xgb_w, lstm_w in WEIGHT_RATIOS:
            combo_probs = xgb_w * xgb_p + lstm_w * lstm_p

            for thresh in CONFIDENCE_THRESHOLDS:
                # High confidence: combined prob > threshold OR < (1-threshold)
                high_conf_buy = combo_probs >= thresh
                high_conf_sell = combo_probs <= (1.0 - thresh)
                high_conf = high_conf_buy | high_conf_sell

                if high_conf.sum() > 0:
                    conf_preds = (combo_probs[high_conf] >= 0.5).astype(int)
                    conf_acc = accuracy_score(y_matched[high_conf], conf_preds)
                    signal_rate = high_conf.sum() / len(high_conf) * 100
                    n_signals = int(high_conf.sum())
                else:
                    conf_acc = None
                    signal_rate = 0.0
                    n_signals = 0

                results[(thresh, xgb_w)].append({
                    'split': split + 1,
                    'accuracy': conf_acc,
                    'signal_rate': signal_rate,
                    'signals': n_signals,
                    'test_size': min_len,
                    'lstm_available': True,
                })

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY: {timeframe}")
    print(f"{'='*70}")

    # XGBoost only baseline
    xgb_accs = [r['accuracy'] for r in xgb_only_results]
    print(f"\n  XGBoost Only (baseline):")
    print(f"    Average: {np.mean(xgb_accs):.2%}")
    print(f"    Min:     {np.min(xgb_accs):.2%}")
    print(f"    Max:     {np.max(xgb_accs):.2%}")
    print(f"    Std:     {np.std(xgb_accs):.2%}")
    for r in xgb_only_results:
        print(f"    Split {r['split']}: {r['accuracy']:.2%} ({r['test_size']} samples)")

    # Ensemble results
    print(f"\n  {'─'*66}")
    print(f"  ENSEMBLE RESULTS (Confidence Threshold Strategy)")
    print(f"  {'─'*66}")

    best_combo = None
    best_avg_acc = 0

    for xgb_w, lstm_w in WEIGHT_RATIOS:
        for thresh in CONFIDENCE_THRESHOLDS:
            split_results = results[(thresh, xgb_w)]
            valid_results = [r for r in split_results if r['accuracy'] is not None]

            if not valid_results:
                continue

            accs = [r['accuracy'] for r in valid_results]
            signal_rates = [r['signal_rate'] for r in valid_results]
            avg_acc = np.mean(accs)
            avg_signal_rate = np.mean(signal_rates)

            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                best_combo = (thresh, xgb_w, lstm_w, avg_acc, np.std(accs),
                              avg_signal_rate, valid_results)

            print(f"\n  Weight {xgb_w:.0%} XGB / {lstm_w:.0%} LSTM | Threshold {thresh}:")
            print(f"    Avg accuracy:    {avg_acc:.2%}")
            print(f"    Std deviation:   {np.std(accs):.2%}")
            print(f"    Avg signal rate: {avg_signal_rate:.1f}%")
            for r in valid_results:
                lstm_flag = "" if r['lstm_available'] else " (XGB only)"
                print(f"    Split {r['split']}: {r['accuracy']:.2%} "
                      f"({r['signals']}/{r['test_size']} signals, "
                      f"{r['signal_rate']:.1f}%){lstm_flag}")

    # Best combination
    if best_combo:
        thresh, xgb_w, lstm_w, avg_acc, std_acc, avg_sr, splits = best_combo
        print(f"\n{'='*70}")
        print(f"BEST ENSEMBLE CONFIGURATION: {timeframe}")
        print(f"{'='*70}")
        print(f"  Weight:          {xgb_w:.0%} XGB / {lstm_w:.0%} LSTM")
        print(f"  Threshold:       {thresh}")
        print(f"  Walk-forward avg: {avg_acc:.2%}")
        print(f"  Std deviation:    {std_acc:.2%}")
        print(f"  Signal rate:      {avg_sr:.1f}%")
        print(f"  XGBoost only avg: {np.mean(xgb_accs):.2%}")
        print(f"  Ensemble lift:    {avg_acc - np.mean(xgb_accs):+.2%}")

        print(f"\n  Per-split breakdown:")
        for r in splits:
            print(f"    Split {r['split']}: {r['accuracy']:.2%} "
                  f"({r['signals']} signals out of {r['test_size']})")

    return results, xgb_only_results, best_combo


if __name__ == "__main__":
    all_results = {}

    for tf in ["_2w", "_30d"]:
        results, xgb_results, best = walk_forward_ensemble(tf, n_splits=5)
        all_results[tf] = {
            'ensemble_results': results,
            'xgb_results': xgb_results,
            'best_combo': best,
        }

    # Final comparison
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON — BOTH TIMEFRAMES")
    print(f"{'='*70}")

    for tf in ["_2w", "_30d"]:
        best = all_results[tf]['best_combo']
        xgb_accs = [r['accuracy'] for r in all_results[tf]['xgb_results']]

        if best:
            thresh, xgb_w, lstm_w, avg_acc, std_acc, avg_sr, _ = best
            print(f"\n  {tf}:")
            print(f"    XGBoost only walk-forward:  {np.mean(xgb_accs):.2%}")
            print(f"    Best ensemble walk-forward: {avg_acc:.2%} "
                  f"({xgb_w:.0%}/{lstm_w:.0%} @ {thresh} threshold)")
            print(f"    Signal rate:                {avg_sr:.1f}%")
            print(f"    Ensemble lift:              {avg_acc - np.mean(xgb_accs):+.2%}")