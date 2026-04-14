import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib


BEST_PARAMS = {
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


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, lstm_output):
        weights = self.attention(lstm_output)
        weights = torch.softmax(weights, dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context


class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional=True):
        super(StockLSTM, self).__init__()

        self.bidirectional = bidirectional
        direction_multiplier = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        self.attention = Attention(hidden_size * direction_multiplier)

        fc_input = hidden_size * direction_multiplier
        self.fc = nn.Sequential(
            nn.Linear(fc_input, fc_input // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input // 2, fc_input // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_input // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        prediction = self.fc(context)
        return prediction


def load_and_prepare_sequences(timeframe="_2w"):
    params = BEST_PARAMS.get(timeframe, BEST_PARAMS["_2w"])
    sequence_length = params['sequence_length']

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

    df_train = df_train.sort_values(['symbol', 'date']).reset_index(drop=True)

    X_raw = df_train[feature_cols].fillna(0).values
    y_raw = df_train['target'].values
    symbols = df_train['symbol'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

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

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    print(f"Built {len(X_sequences)} sequences of length {sequence_length}")
    print(f"  Positive (buy): {y_sequences.sum():.0f}")
    print(f"  Negative (avoid): {len(y_sequences) - y_sequences.sum():.0f}")

    split_point = int(len(X_sequences) * 0.8)
    X_train = X_sequences[:split_point]
    y_train = y_sequences[:split_point]
    X_test = X_sequences[split_point:]
    y_test = y_sequences[split_point:]

    print(f"\nTraining set: {len(X_train)} sequences")
    print(f"Test set: {len(X_test)} sequences")

    return X_train, y_train, X_test, y_test, feature_cols, scaler


def train_lstm(X_train, y_train, X_test, y_test, feature_cols, timeframe="_2w"):
    params = BEST_PARAMS.get(timeframe, BEST_PARAMS["_2w"])
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

    # Learning rate scheduler - reduce LR when accuracy plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    train_dataset = StockDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True)

    test_dataset = StockDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False, drop_last=True)

    epochs = 100
    print(f"\nTraining Enhanced LSTM model...")
    print(f"  Bidirectional + Attention + LR Scheduler")
    print(f"  Hidden: {params['hidden_size']}, Layers: {params['num_layers']}, Dropout: {params['dropout']}")
    print(f"  LR: {params['learning_rate']}, Batch: {params['batch_size']}")
    print(f"  Epochs: {epochs}")

    best_accuracy = 0
    best_model_state = None
    no_improve_count = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(-1)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        if (epoch + 1) % 5 == 0:
            model.eval()
            all_preds = []
            all_true = []

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X).squeeze(-1)
                    preds = (outputs >= 0.5).float()
                    all_preds.extend(preds.numpy())
                    all_true.extend(batch_y.numpy())

            accuracy = accuracy_score(all_true, all_preds)
            scheduler.step(accuracy)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%} | LR: {current_lr:.6f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_state = model.state_dict().copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Early stopping if no improvement for 20 evaluation cycles
            if no_improve_count >= 8:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)

    model.eval()
    all_preds = []
    all_true = []
    all_probs = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X).squeeze(-1)
            preds = (outputs >= 0.5).float()
            all_preds.extend(preds.numpy())
            all_true.extend(batch_y.numpy())
            all_probs.extend(outputs.numpy())

    accuracy = accuracy_score(all_true, all_preds)
    print(f"\nBest LSTM Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=['Avoid', 'Buy']))

    return model, best_accuracy, all_probs, all_true


def save_lstm_model(model, scaler, timeframe=""):
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f'lstm{timeframe}_{timestamp}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\nLSTM model saved to {model_path}")

    scaler_path = os.path.join(model_dir, f'lstm_scaler{timeframe}_{timestamp}.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    for tf in ["_2w", "_30d"]:
        print(f"\n{'='*50}")
        print(f"TRAINING LSTM {tf} MODEL")
        print(f"{'='*50}")
        X_train, y_train, X_test, y_test, feature_cols, scaler = load_and_prepare_sequences(tf)
        model, accuracy, probs, true_labels = train_lstm(X_train, y_train, X_test, y_test, feature_cols, tf)
        save_lstm_model(model, scaler, tf)