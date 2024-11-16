from dotenv import load_dotenv
import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None

    def prepare_data(self, df, sequence_length=60):
        """Prepare data for LSTM model"""
        # Select relevant features
        data = df[['close', 'volume', 'high', 'low']].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
            y.append(scaled_data[i + sequence_length, 0])
            
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_model(self, df, sequence_length=60):
        """Train the prediction model"""
        if df is None or len(df) < sequence_length:
            return False
            
        X, y = self.prepare_data(df, sequence_length)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Build and train model
        self.model = self.build_model(input_shape=(X.shape[1], X.shape[2]))
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        
        return True

    def predict_future(self, df, days_ahead=30):
        """Predict future stock prices"""
        if self.model is None:
            self.train_model(df)
            
        last_sequence = df[['close', 'volume', 'high', 'low']].values[-60:]
        last_sequence = self.scaler.transform(last_sequence)
        
        future_predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days_ahead):
            next_day = self.model.predict(current_sequence.reshape(1, 60, 4), verbose=0)
            next_row = current_sequence[-1].copy()
            next_row[0] = next_day[0][0]
            
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_row
            
            future_predictions.append(next_day[0][0])
        
        dummy_array = np.repeat(current_sequence[-1:], len(future_predictions), axis=0)
        dummy_array[:, 0] = future_predictions
        future_predictions = self.scaler.inverse_transform(dummy_array)[:, 0]
        
        return future_predictions
