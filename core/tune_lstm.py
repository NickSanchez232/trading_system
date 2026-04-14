import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import optuna
from datetime import datetime

optuna.logging.set_verbosity(optuna.logging.WARNING)


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StockLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction


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

    X_raw = df_train[feature_cols].fillna(0).values
    y_raw = df_train['target'].values
    symbols = df_train['symbol'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return X_scaled, y_raw, symbols, feature_cols, scaler


def build_sequences(X_scaled, y_raw, symbols, sequence_length):
    X_sequences = []
    y_sequences = []

    unique_symbols = np.unique(symbols)
    for symbol in unique_symbols:
        mask = symbols == symbol
        X_symbol = X_scaled[mask]
        y_symbol = y_raw[mask]

        for i in range(sequence_length, len(X_symbol)):
            X_sequences.append(X_symbol[i - sequence_length:i])
            y_sequences.append(y_symbol[i])

    return np.array(X_sequences), np.array(y_sequences)


def objective(trial, X_scaled, y_raw, symbols, feature_cols):
    try:
        # Parameters to tune
        sequence_length = trial.suggest_int('sequence_length', 3, 30)
        hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128, 256, 512])
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.05, 0.6)
        learning_rate = trial.suggest_float('learning_rate', 0.00005, 0.05, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
        epochs = 40

        # Build sequences
        X_seq, y_seq = build_sequences(X_scaled, y_raw, symbols, sequence_length)

        # Split
        split_point = int(len(X_seq) * 0.8)
        X_train = X_seq[:split_point]
        y_train = y_seq[:split_point]
        X_test = X_seq[split_point:]
        y_test = y_seq[split_point:]

        if len(X_test) == 0:
            return 0.5

        # Build model
        input_size = len(feature_cols)
        model = StockLSTM(input_size, hidden_size, num_layers, dropout)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        train_dataset = StockDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        test_dataset = StockDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        # Train
        best_accuracy = 0
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze(-1)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Evaluate
            model.eval()
            all_preds = []
            all_true = []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X).squeeze(-1)
                    preds = (outputs >= 0.5).float()
                    all_preds.extend(preds.numpy())
                    all_true.extend(batch_y.numpy())

            if len(all_preds) == 0:
                return 0.5

            accuracy = accuracy_score(all_true, all_preds)
            if accuracy > best_accuracy:
                best_accuracy = accuracy

            # Early stopping - prune bad trials
            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        print(f"  Trial {trial.number}: seq={sequence_length}, hidden={hidden_size}, "
              f"layers={num_layers}, dropout={dropout:.2f}, lr={learning_rate:.5f}, "
              f"batch={batch_size}, wd={weight_decay:.6f} -> {best_accuracy:.2%}")

        return best_accuracy

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        return 0.5


if __name__ == "__main__":
    for tf in ["_2w", "_30d"]:
        print(f"\n{'='*60}")
        print(f"TUNING LSTM {tf} MODEL - 75 TRIALS")
        print(f"{'='*60}")

        X_scaled, y_raw, symbols, feature_cols, scaler = load_data(tf)
        print(f"Loaded data: {len(X_scaled)} rows, {len(feature_cols)} features")

        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        study.optimize(
            lambda trial: objective(trial, X_scaled, y_raw, symbols, feature_cols),
            n_trials=75,
        )

        print(f"\n{'='*60}")
        print(f"BEST LSTM PARAMETERS {tf}")
        print(f"{'='*60}")
        print(f"  Best accuracy: {study.best_value:.2%}")
        print(f"  Best parameters:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")