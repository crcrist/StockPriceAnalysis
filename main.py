import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
import time
from database import StockDatabase
from data_collector import DataCollector, test_single_stock_api
from stock_predictor import StockPredictor

def update_database(limit=None, symbols=None):
    try:
        print("Starting database update...")
        db = StockDatabase()
        collector = DataCollector()
        predictor = StockPredictor()
        
        # Check API status first
        api_status = collector.check_api_status()
        if api_status['status'] != "API Connected":
            st.error(f"API Status: {api_status['status']}")
            return
        
        st.info(f"API Status: {api_status['status']}")
        
        print("Fetching stock list...")
        stocks = collector.get_all_stocks()
        
        if not stocks:
            st.error("No valid stocks found. Please check the API connection and filters.")
            return
            
        # Apply limits or filter by symbols if specified
        if limit:
            stocks = stocks[:limit]
        elif symbols:
            stocks = [s for s in stocks if s['symbol'] in symbols]
            
        total_stocks = len(stocks)
        st.info(f"Found {total_stocks} stocks to process")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_text = st.empty()
        
        processed = 0
        successful = 0
        failed = 0
        
        # Process stocks in batches
        batch_size = 10
        for i in range(0, len(stocks), batch_size):
            batch = stocks[i:i + batch_size]
            batch_symbols = [stock['symbol'] for stock in batch]
            
            status_text.text(f"Processing batch {i//batch_size + 1}/{(total_stocks + batch_size - 1)//batch_size}")
            
            # Fetch historical data for batch
            historical_data = collector.get_historical_data_batch(batch_symbols, '2024-01-01')
            
            # Process each stock in the batch
            for stock in batch:
                symbol = stock['symbol']
                try:
                    processed += 1
                    db.insert_stock(symbol, stock.get('name', ''), stock.get('exchangeShortName', ''))
                    
                    if symbol in historical_data:
                        hist_data = historical_data[symbol]
                        if not hist_data.empty:
                            db.insert_historical_data(symbol, hist_data)
                            
                            future_prices = predictor.predict_future(hist_data)
                            if future_prices is not None:
                                future_dates = [datetime.now() + timedelta(days=x) 
                                              for x in range(1, 31)]
                                predictions_df = pd.DataFrame({
                                    'prediction_date': future_dates,
                                    'predicted_price': future_prices
                                })
                                db.insert_predictions(symbol, predictions_df)
                                successful += 1
                            else:
                                failed += 1
                        else:
                            failed += 1
                            print(f"Empty historical data for {symbol}")
                    else:
                        failed += 1
                        print(f"No historical data received for {symbol}")
                        
                except Exception as e:
                    failed += 1
                    print(f"Error processing {symbol}: {str(e)}")
                    continue
                
                # Update stats
                stats_text.text(f"""
                Processed: {processed}/{total_stocks}
                Successful: {successful}
                Failed: {failed}
                """)
            
            # Update progress
            progress = (i + len(batch)) / total_stocks
            progress_bar.progress(progress)
        
        status_text.text("Database update complete!")
        progress_bar.progress(1.0)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        raise e

def selective_update():
    st.sidebar.markdown("### Update Options")
    
    # Test API first
    api_working, message = test_single_stock_api()
    
    update_options = st.sidebar.radio(
        "Choose update type:",
        ["Full Update", "Top Stocks Only", "Selected Stocks", "Check API Status Only"]
    )
    
    collector = DataCollector()
    api_status = collector.check_api_status()
    
    st.sidebar.info(f"""
    API Status: {api_status['status']}
    Status: {api_status['calls_remaining']}
    Reset: {api_status['time_until_reset']}
    """)
    
    if not api_working:
        st.sidebar.error(f"API Test Failed: {message}")
        return
        
    if update_options == "Full Update":
        if st.sidebar.button("Update All Stocks"):
            update_database()
            
    elif update_options == "Top Stocks Only":
        num_stocks = st.sidebar.slider("Number of stocks to update", 10, 100, 25)
        if st.sidebar.button(f"Update Top {num_stocks} Stocks"):
            update_database(limit=num_stocks)
            
    elif update_options == "Selected Stocks":
        db = StockDatabase()
        available_symbols = pd.read_sql_query(
            "SELECT DISTINCT symbol FROM stocks ORDER BY symbol", 
            db.conn
        )['symbol'].tolist()
        
        selected_symbols = st.sidebar.multiselect(
            "Select stocks to update",
            available_symbols
        )
        
        if st.sidebar.button("Update Selected Stocks") and selected_symbols:
            update_database(symbols=selected_symbols)
            
    elif update_options == "Check API Status Only":
        st.sidebar.success(f"API Status: {api_status['status']}")

def create_dashboard():
    st.title("US Stock Market Analytics Dashboard")
    
    db = StockDatabase()
    
    # Sidebar for stock selection
    query = "SELECT DISTINCT symbol FROM stocks ORDER BY symbol"
    symbols = pd.read_sql_query(query, db.conn)['symbol'].tolist()
    
    if not symbols:
        st.warning("No stock data available. Please click 'Update Database' to fetch data.")
        return
        
    selected_symbol = st.sidebar.selectbox("Select Stock", symbols)
    
    if selected_symbol:
        # Get historical data
        hist_data = db.get_stock_performance(selected_symbol)
        
        if hist_data.empty:
            st.warning(f"No historical data available for {selected_symbol}")
            return
            
        # Get predictions
        pred_query = '''
            SELECT prediction_date, predicted_price
            FROM predictions
            WHERE symbol = ?
            AND prediction_made_at = (
                SELECT MAX(prediction_made_at)
                FROM predictions
                WHERE symbol = ?
            )
        '''
        predictions = pd.read_sql_query(pred_query, db.conn, 
                                      params=(selected_symbol, selected_symbol))
        
        # Create plot
        fig = go.Figure()
        
        # Historical prices
        fig.add_trace(go.Scatter(
            x=hist_data['date'],
            y=hist_data['close'],
            name='Historical Price',
            line=dict(color='blue')
        ))
        
        # Predictions
        if not predictions.empty:
            fig.add_trace(go.Scatter(
                x=predictions['prediction_date'],
                y=predictions['predicted_price'],
                name='Predicted Price',
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            title=f"{selected_symbol} Stock Price",
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig)
        
        # Performance metrics
        if not hist_data.empty:
            start_price = hist_data.iloc[0]['close']
            current_price = hist_data.iloc[-1]['close']
            perf = ((current_price - start_price) / start_price) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("YTD Performance", f"{perf:.2f}%")
            col2.metric("Start Price", f"${start_price:.2f}")
            col3.metric("Current Price", f"${current_price:.2f}")
            
            # Add more analysis metrics
            if len(hist_data) > 1:
                # Calculate daily returns
                hist_data['daily_return'] = hist_data['close'].pct_change()
                
                # Calculate metrics
                volatility = hist_data['daily_return'].std() * (252 ** 0.5) * 100
                max_drawdown = ((hist_data['close'].cummax() - hist_data['close']) / hist_data['close'].cummax()).max() * 100
                
                # Display additional metrics
                col1, col2 = st.columns(2)
                col1.metric("Annualized Volatility", f"{volatility:.2f}%")
                col2.metric("Maximum Drawdown", f"{max_drawdown:.2f}%")
                
                # Add description
                st.markdown("""
                **Metrics Explanation:**
                - **YTD Performance**: Year-to-date price change percentage
                - **Annualized Volatility**: A measure of price variation (risk) on a yearly basis
                - **Maximum Drawdown**: Largest peak-to-trough decline percentage
                """)

if __name__ == "__main__":
    selective_update()
    create_dashboard()
