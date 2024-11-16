# data_collector.py
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime, timedelta
import pandas as pd
import requests
from dotenv import load_dotenv
import os

class DataCollector:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('FMP_API_KEY')
        if not self.api_key:
            raise ValueError("API key not found in environment variables")
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.session = requests.Session()

    def check_api_status(self):
        """Check API quota status"""
        endpoint = f"{self.base_url}/stock/list"  # Using a different endpoint to test API
        params = {'apikey': self.api_key}
        
        try:
            response = self.session.get(endpoint, params=params)
            if response.status_code == 200:
                return {
                    'calls_remaining': "Available",
                    'status': "API Connected",
                    'time_until_reset': "24 hours"
                }
            elif response.status_code == 429:  # Too Many Requests
                return {
                    'calls_remaining': "0",
                    'status': "Quota Exceeded",
                    'time_until_reset': "24 hours"
                }
            else:
                return {
                    'calls_remaining': "Unknown",
                    'status': f"Error: {response.status_code}",
                    'time_until_reset': "Unknown"
                }
        except Exception as e:
            print(f"Error checking API status: {str(e)}")
            return {
                'calls_remaining': "Error",
                'status': "Connection Failed",
                'time_until_reset': "Unknown"
            }
    
    def is_valid_stock(self, stock):
        """Check if a stock entry is valid for processing"""
        try:
            # Check basic requirements
            if not stock.get('symbol'):
                return False
                
            # Check exchange
            if stock.get('exchangeShortName') not in ['NYSE', 'NASDAQ', 'AMEX']:
                return False
                
            # Check symbol format
            if any(x in stock.get('symbol', '') for x in ['^', '/', '.', '$']):
                return False
                
            # Check type
            if stock.get('type') != 'stock':
                return False
                
            # Check price (safely)
            price_str = stock.get('price')
            if price_str is None:
                return False
            try:
                price = float(price_str)
                if price <= 0:
                    return False
            except (ValueError, TypeError):
                return False
                
            return True
            
        except Exception as e:
            print(f"Error validating stock {stock.get('symbol', 'UNKNOWN')}: {str(e)}")
            return False

    def get_all_stocks(self):
        """Fetch all US stock symbols with filtering"""
        endpoint = f"{self.base_url}/stock/list"
        params = {'apikey': self.api_key}
        
        try:
            response = self.session.get(endpoint, params=params)
            if response.status_code == 200:
                stocks = response.json()
                # Filter valid stocks
                us_stocks = [s for s in stocks if self.is_valid_stock(s)]
                print(f"Found {len(us_stocks)} valid stocks out of {len(stocks)} total")
                return us_stocks[:100]  # Limit for testing - remove for production
            else:
                print(f"API request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error fetching stock list: {str(e)}")
        return []

    def get_historical_data_batch(self, symbols, start_date):
        """Fetch historical data for multiple symbols in parallel"""
        def fetch_single_stock(symbol):
            endpoint = f"{self.base_url}/historical-price-full/{symbol}"
            params = {
                'apikey': self.api_key,
                'from': start_date,
                'to': datetime.now().strftime('%Y-%m-%d')
            }
            
            try:
                response = self.session.get(endpoint, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if 'historical' in data:
                        df = pd.DataFrame(data['historical'])
                        df['symbol'] = symbol
                        return symbol, df
                else:
                    print(f"Failed to fetch {symbol}: Status {response.status_code}")
            except Exception as e:
                print(f"Error fetching {symbol}: {str(e)}")
            return symbol, None

        results = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {executor.submit(fetch_single_stock, symbol): symbol 
                              for symbol in symbols}
            
            for future in as_completed(future_to_symbol):
                symbol, df = future.result()
                if df is not None:
                    results[symbol] = df
                time.sleep(0.2)  # Gentle rate limiting
                
        return results

def test_single_stock_api():
    """Test API with a single stock request"""
    collector = DataCollector()
    test_symbol = "AAPL"  # Using Apple as a test case
    
    try:
        endpoint = f"{collector.base_url}/quote/{test_symbol}"
        params = {'apikey': collector.api_key}
        
        response = collector.session.get(endpoint, params=params)
        
        if response.status_code == 200:
            return True, "API working correctly"
        elif response.status_code == 429:
            return False, "API quota exceeded"
        else:
            return False, f"API error: {response.status_code}"
            
    except Exception as e:
        return False, f"Connection error: {str(e)}"
