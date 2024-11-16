import sqlite3
from datetime import datetime
import pandas as pd

class StockDatabase:
    def __init__(self, db_name='stock_analytics.db'):
        self.db_name = db_name
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            # Stocks table for basic company info
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS stocks (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    exchange TEXT,
                    last_updated TIMESTAMP
                )
            ''')
            
            # Historical data table with all columns from API
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS historical_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    unadjusted_volume INTEGER,
                    change REAL,
                    change_percent REAL,
                    vwap REAL,
                    label TEXT,
                    change_over_time REAL,
                    UNIQUE(symbol, date),
                    FOREIGN KEY (symbol) REFERENCES stocks(symbol)
                )
            ''')
            
            # Predictions table
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    prediction_date DATE,
                    predicted_price REAL,
                    prediction_made_at TIMESTAMP,
                    FOREIGN KEY (symbol) REFERENCES stocks(symbol)
                )
            ''')

    def is_stock_up_to_date(self, symbol):
        """Check if stock data is already up to date"""
        query = '''
            SELECT MAX(date) as last_update
            FROM historical_data
            WHERE symbol = ?
        '''
        result = pd.read_sql_query(query, self.conn, params=(symbol,))
        if result['last_update'].iloc[0] is not None:
            last_update = pd.to_datetime(result['last_update'].iloc[0])
            return (datetime.now() - last_update).days < 1
        return False

    def insert_stock(self, symbol, name, exchange):
        with self.conn:
            self.conn.execute('''
                INSERT OR REPLACE INTO stocks (symbol, name, exchange, last_updated)
                VALUES (?, ?, ?, ?)
            ''', (symbol, name, exchange, datetime.now()))

    def insert_historical_data(self, symbol, data_df):
        """Insert historical data with column name mapping"""
        if data_df is None or data_df.empty:
            return
            
        # Rename columns to match database schema
        column_mapping = {
            'adjClose': 'adj_close',
            'unadjustedVolume': 'unadjusted_volume',
            'changePercent': 'change_percent',
            'changeOverTime': 'change_over_time'
        }
        
        # Create a copy of the dataframe to avoid modifying the original
        df_to_insert = data_df.copy()
        
        # Rename columns
        for old_col, new_col in column_mapping.items():
            if old_col in df_to_insert.columns:
                df_to_insert = df_to_insert.rename(columns={old_col: new_col})
        
        # Add symbol column
        df_to_insert['symbol'] = symbol
        
        # Insert data
        try:
            df_to_insert.to_sql('historical_data', self.conn, 
                               if_exists='append', index=False)
        except Exception as e:
            print(f"Error inserting data for {symbol}: {str(e)}")
            print(f"Columns in dataframe: {df_to_insert.columns.tolist()}")

    def insert_predictions(self, symbol, predictions_df):
        predictions_df['symbol'] = symbol
        predictions_df['prediction_made_at'] = datetime.now()
        predictions_df.to_sql('predictions', self.conn, if_exists='append', index=False)

    def get_stock_performance(self, symbol, start_date='2024-01-01'):
        query = '''
            SELECT date, close
            FROM historical_data
            WHERE symbol = ? AND date >= ?
            ORDER BY date
        '''
        return pd.read_sql_query(query, self.conn, params=(symbol, start_date))
