import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from datetime import datetime


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(StockLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction


def load_and_prepare_sequences(timeframe="_2w", sequence_length=10):
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'training_dataset.csv')
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} rows")

    df = df[df['timeframe'] == timeframe]
    print(f"Filtered to {timeframe}: {len(df)} rows")

    df_train = df[df['label'].isin(['strong_positive', 'strong_negative'])].copy()
    print(f"Strong signals only: {len(df_train)} rows")

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
    print(f"Features: {len(feature_cols)}")

    # Sort by symbol then date for proper sequences
    df_train = df_train.sort_values(['symbol', 'date']).reset_index(drop=True)

    # Fill NaN with 0 for LSTM (can't handle NaN)
    X_raw = df_train[feature_cols].fillna(0).values
    y_raw = df_train['target'].values
    symbols = df_train['symbol'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Build sequences - group by symbol so we don't mix stocks
    X_sequences = []
    y_sequences = []

    unique_symbols = df_train['symbol'].unique()
    for symbol in unique_symbols:
        mask = symbols == symbol
        X_symbol = X_scaled[mask]
        y_symbol = y_raw[mask]

        for i in range(sequence_length, len(X_symbol)):
            X_sequences.append(X_symbol[i - sequence_length:i])
            y_sequences.append(y_symbol[i])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    print(f"Built {len(X_sequences)} sequences of length {sequence_length}")
    print(f"  Positive (buy): {y_sequences.sum():.0f}")
    print(f"  Negative (avoid): {len(y_sequences) - y_sequences.sum():.0f}")

    # Split by time (80/20)
    split_point = int(len(X_sequences) * 0.8)
    X_train = X_sequences[:split_point]
    y_train = y_sequences[:split_point]
    X_test = X_sequences[split_point:]
    y_test = y_sequences[split_point:]

    print(f"\nTraining set: {len(X_train)} sequences")
    print(f"Test set: {len(X_test)} sequences")

    return X_train, y_train, X_test, y_test, feature_cols, scaler


def train_lstm(X_train, y_train, X_test, y_test, feature_cols, epochs=50, batch_size=256, learning_rate=0.001):
    input_size = len(feature_cols)

    model = StockLSTM(input_size=input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = StockDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = StockDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nTraining LSTM model...")
    print(f"  Architecture: {input_size} input -> 128 hidden -> 64 -> 32 -> 1 output")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")

    best_accuracy = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            all_preds = []
            all_true = []

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X).squeeze()
                    preds = (outputs >= 0.5).float()
                    all_preds.extend(preds.numpy())
                    all_true.extend(batch_y.numpy())

            accuracy = accuracy_score(all_true, all_preds)
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2%}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict().copy()

    # Load best model
    model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    all_preds = []
    all_true = []
    all_probs = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X).squeeze()
            preds = (outputs >= 0.5).float()
            all_preds.extend(preds.numpy())
            all_true.extend(batch_y.numpy())
            all_probs.extend(outputs.numpy())

    accuracy = accuracy_score(all_true, all_preds)
    print(f"\nBest LSTM Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=['Avoid', 'Buy']))

    return model, best_accuracy


def save_lstm_model(model, scaler, timeframe=""):
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f'lstm{timeframe}_{timestamp}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\nLSTM model saved to {model_path}")

    scaler_path = os.path.join(model_dir, f'lstm_scaler{timeframe}_{timestamp}.pkl')
    import joblib
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    for tf in ["_2w", "_30d"]:
        print(f"\n{'='*50}")
        print(f"TRAINING LSTM {tf} MODEL")
        print(f"{'='*50}")
        X_train, y_train, X_test, y_test, feature_cols, scaler = load_and_prepare_sequences(tf, sequence_length=10)
        model, accuracy = train_lstm(X_train, y_train, X_test, y_test, feature_cols)
        save_lstm_model(model, scaler, tf)