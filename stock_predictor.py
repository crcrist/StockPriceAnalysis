import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import ta  # Technical Analysis library

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataset"""
        # Make sure we have a copy with correct column names
        df = df.copy()
        if 'adj_close' in df.columns:
            close_col = 'adj_close'
        else:
            close_col = 'close'
            
        # Add RSI
        df['rsi'] = ta.momentum.RSIIndicator(df[close_col]).rsi()
        
        # Add MACD
        macd = ta.trend.MACD(df[close_col])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # Add Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df[close_col])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        
        # Add Moving Averages
        df['sma_20'] = ta.trend.SMAIndicator(df[close_col], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(df[close_col], window=50).sma_indicator()
        
        # Add Average True Range (ATR)
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df[close_col]).average_true_range()
        
        # Drop any NaN values that resulted from indicator calculations
        df = df.dropna()
        
        return df
        
    def prepare_data(self, df, sequence_length=60):
        """Prepare data with technical indicators for LSTM model"""
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Select features for model
        feature_columns = [
            'close', 'volume', 'high', 'low',
            'rsi', 'macd', 'macd_signal',
            'bb_high', 'bb_low',
            'sma_20', 'sma_50', 'atr'
        ]
        
        data = df[feature_columns].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
            y.append(scaled_data[i + sequence_length, 0])  # Predicting close price
            
        return np.array(X), np.array(y)
        
    def build_model(self, input_shape):
        """Build an improved LSTM model"""
        model = Sequential([
            # First Bidirectional LSTM layer
            Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape),
            Dropout(0.3),
            
            # Second Bidirectional LSTM layer
            Bidirectional(LSTM(100, return_sequences=True)),
            Dropout(0.3),
            
            # Third Bidirectional LSTM layer
            Bidirectional(LSTM(100, return_sequences=False)),
            Dropout(0.3),
            
            # Dense layers
            Dense(100, activation='relu'),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber'  # Huber loss is more robust to outliers
        )
        
        return model
        
    def train_model(self, df, sequence_length=60):
        """Train the prediction model with validation"""
        if df is None or len(df) < sequence_length:
            return False
            
        X, y = self.prepare_data(df, sequence_length)
        
        # Use the last 20% as validation data
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Build and train model
        self.model = self.build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Early stopping to prevent overfitting
        from tensorflow.keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        return True
        
    def predict_future(self, df, days_ahead=30):
        """Predict future stock prices with trend analysis"""
        if self.model is None:
            self.train_model(df)
            
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Get the last sequence
        feature_columns = [
            'close', 'volume', 'high', 'low',
            'rsi', 'macd', 'macd_signal',
            'bb_high', 'bb_low',
            'sma_20', 'sma_50', 'atr'
        ]
        
        last_sequence = df[feature_columns].values[-60:]
        last_sequence = self.scaler.transform(last_sequence)
        
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        # Get recent trend
        recent_trend = np.mean(np.diff(df['close'].values[-10:]))
        
        for _ in range(days_ahead):
            # Predict next day
            next_day = self.model.predict(current_sequence.reshape(1, 60, len(feature_columns)), verbose=0)
            
            # Adjust prediction based on recent trend
            trend_adjustment = recent_trend * 0.1  # Reduce trend impact
            next_day[0][0] += trend_adjustment
            
            # Create a dummy row for the next day
            next_row = current_sequence[-1].copy()
            next_row[0] = next_day[0][0]
            
            # Update other features (simple approach)
            next_row[1:] = current_sequence[-1, 1:]  # Copy previous day's values
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_row
            
            # Store prediction
            future_predictions.append(next_day[0][0])
        
        # Inverse transform predictions
        dummy_array = np.repeat(current_sequence[-1:], len(future_predictions), axis=0)
        dummy_array[:, 0] = future_predictions
        future_predictions = self.scaler.inverse_transform(dummy_array)[:, 0]
        
        return future_predictions
