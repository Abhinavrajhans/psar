import pandas as pd 
import calendar
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import calendar
import scipy.stats as si
# utilities.py
# utilities.py
import warnings
# Suppress all warnings globally
warnings.filterwarnings("ignore")
import pandas as pd
from multiprocessing import Pool, Manager
from datetime import timedelta

def calculate_sar(df, acceleration=0.02, maximum=0.2):
    """
    Calculate Parabolic SAR for a given dataframe containing price data.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with columns 'EQ_High' and 'EQ_Low'
    acceleration (float): Starting acceleration factor (default: 0.02)
    maximum (float): Maximum acceleration factor (default: 0.2)
    
    Returns:
    dict: Dictionary containing 'SAR' and 'Trend' Series
    """
    # First rename the columns
    df = df.rename(columns={
        'EQ_Open': 'Open',
        'EQ_High': 'High',
        'EQ_Low': 'Low',
        'EQ_Close': 'Close'
    })
    
    high = df['High']
    low = df['Low']
    
    # Initialize arrays
    sar = np.zeros(len(df))
    ep = np.zeros(len(df))  # Extreme point
    af = np.zeros(len(df))  # Acceleration factor
    trend = np.zeros(len(df))  # 1 for uptrend, -1 for downtrend
    
    # Initialize first values
    trend[0] = 1  # Assume uptrend to start
    sar[0] = low[0]  # Start with first low for uptrend
    ep[0] = high[0]  # First extreme point
    af[0] = acceleration
    
    # Calculate SAR values
    for i in range(1, len(df)):
        # Previous values
        sar_prev = sar[i-1]
        ep_prev = ep[i-1]
        af_prev = af[i-1]
        trend_prev = trend[i-1]
        
        # Calculate SAR for current period
        sar[i] = sar_prev + af_prev * (ep_prev - sar_prev)
        
        # Update trend
        if trend_prev == 1:  # Previous uptrend
            trend[i] = 1
            if low[i] < sar[i]:  # Trend reversal
                trend[i] = -1
                sar[i] = ep_prev
                ep[i] = low[i]
                af[i] = acceleration
            else:
                if high[i] > ep_prev:  # New high
                    ep[i] = high[i]
                    af[i] = min(af_prev + acceleration, maximum)
                else:
                    ep[i] = ep_prev
                    af[i] = af_prev
                # Ensure SAR is below the recent lows
                sar[i] = min(sar[i], low[i-1], low[i])
                
        else:  # Previous downtrend
            trend[i] = -1
            if high[i] > sar[i]:  # Trend reversal
                trend[i] = 1
                sar[i] = ep_prev
                ep[i] = high[i]
                af[i] = acceleration
            else:
                if low[i] < ep_prev:  # New low
                    ep[i] = low[i]
                    af[i] = min(af_prev + acceleration, maximum)
                else:
                    ep[i] = ep_prev
                    af[i] = af_prev
                # Ensure SAR is above the recent highs
                sar[i] = max(sar[i], high[i-1], high[i])
    
    # Create Series for both SAR and Trend
    sar_series = pd.Series(sar, index=df.index)
    trend_series = pd.Series(trend, index=df.index)
    
    return {'SAR': sar_series, 'Trend': trend_series}

# Use the function this way:


############# Helper Functions ##################
def last_friday_of_previous_month(year, month):
    if month == 1:
        year -= 1
        month = 12
    else:
        month -= 1

    last_day = calendar.monthrange(year, month)[1]
    last_date = datetime(year, month, last_day)

    offset = (last_date.weekday() - 3) % 7  # 3 = Thursday
    last_thursday_date = last_date - timedelta(days=offset)
    return last_thursday_date + timedelta(days=1)  # Adjusting to Friday if needed

def last_thursday(year, month):
    last_day = calendar.monthrange(year, month)[1]
    last_date = datetime(year, month, last_day)

    offset = (last_date.weekday() - 3) % 7  # 3 = Thursday
    last_thursday_date = last_date - timedelta(days=offset)
    return last_thursday_date

def extract_strike_price_and_type(ticker):
    parts = ticker.split('-')
    strike_and_type = parts[-1]
    strike_price = ''.join([char for char in strike_and_type if char.isdigit() or char == '.'])
    option_type = 'call' if 'CE' in strike_and_type else 'put' if 'PE' in strike_and_type else None
    return float(strike_price), option_type


def calculate_historical_volatility(equity_data, lookback_period=252):
    log_returns = np.log(equity_data['EQ_Close'] / equity_data['EQ_Close'].shift(1))
    rolling_std = log_returns.rolling(window=lookback_period).std()
    volatility = rolling_std * np.sqrt(252)
    volatility = pd.Series(volatility, index=equity_data.index).fillna(0.3)
    return { 'volatility': volatility}


def calculate_time_to_maturity(current_date,lt):
    days_to_maturity = (lt - current_date).days
    return days_to_maturity / 365.0

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        delta = si.norm.cdf(d1)
    elif option_type == 'put':
        delta = -si.norm.cdf(-d1)

    return delta

def find_option_by_delta(options_for_date, spot_price, time_to_maturity, volatility, target_delta, option_type='call'):
    options_for_date = options_for_date[options_for_date['Extracted Option Type'] == option_type]    
    options_for_date.loc[:, 'Calculated_Delta'] = options_for_date.apply(
        lambda row: calculate_greeks(spot_price, row['Strike Price'], time_to_maturity, 0.07, volatility, option_type), axis=1)
    options_for_date.loc[:, 'Delta_Diff'] = abs(options_for_date['Calculated_Delta'] - target_delta)
    if len(options_for_date) == 0:
        return None
    return options_for_date.loc[options_for_date['Delta_Diff'].idxmin()]


def get_option_price(options_data, strike_price, option_type='call', ohlc='Close'):
    option_row = options_data[(options_data['Strike Price'] == strike_price) & (options_data['Extracted Option Type'] == option_type)]
    return option_row[ohlc].values[0] if not option_row.empty else None


def correct_price_on_expiry(strike,spot,option_type):
    if option_type=='call':
        return max(spot-strike,0)
    if option_type=='put':
        return min(strike-spot,0)

import multiprocessing
import pandas as pd
from datetime import timedelta

# Define the function to process each ticker in parallel
def process_ticker(ticker, start_year, end_year, exposure):
    option_trades = []
    stock_data = pd.read_csv(f"./Stocks Data/{ticker}_EQ_EOD.csv")

    # Use the function this way:
    result = calculate_sar(stock_data)
    stock_data['SAR'] = result['SAR']
    stock_data['Trend'] = result['Trend']
    result =calculate_historical_volatility(stock_data)
    stock_data['Volatility'] = result['volatility']
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.strftime('%Y-%m-%d')

    op_data=pd.read_csv(f"./Stocks Data/{ticker}_Opt_EOD.csv")
    op_data['Date'] = pd.to_datetime(op_data['Date']).dt.strftime('%Y-%m-%d')
    op_data[['Strike Price', 'Extracted Option Type']] = op_data['Ticker'].apply(extract_strike_price_and_type).apply(pd.Series)
    for year in range(start_year, end_year):
        for month in range(1, 13):
            lf=last_friday_of_previous_month(year, month)
            lt = last_thursday(year, month)

            lf_str = lf.strftime('%Y-%m-%d')
            lt_str = lt.strftime('%Y-%m-%d')
            #  Entry
            filter_stock_data = stock_data[(stock_data['Date'] >= lf_str) & (stock_data['Date'] <= lt_str)]
            filter_op_data = op_data[(op_data['Date'] >= lf_str) & (op_data['Date'] <= lt_str)]
            

            current_date = lf
            is_option_open=False
            valid_expriy_day=None 
            in_trade_option_type=None
            while current_date <= lt:
                current_date_str=current_date.strftime('%Y-%m-%d')
                stock_data_today=filter_stock_data[filter_stock_data['Date']==current_date_str]
                options_data_today=filter_op_data[filter_op_data['Date']==current_date_str]
                time_to_maturity = calculate_time_to_maturity(current_date,lt)

                if len(stock_data_today)==0  or len(options_data_today)==0:

                    if (current_date==lt) and (valid_expriy_day!=None):
                    
                        valid_expriy_day_str=valid_expriy_day.strftime('%Y-%m-%d')
                        stock_data_today=filter_stock_data[filter_stock_data['Date']==valid_expriy_day_str]
                        options_data_today=filter_op_data[filter_op_data['Date']==valid_expriy_day_str]
                        time_to_maturity = calculate_time_to_maturity(valid_expriy_day,lt)
                        spot=stock_data_today['EQ_Close'].values[0]
                        volatility=stock_data_today['Volatility'].values[0]
                        trend=stock_data_today['Trend'].values[0]
                        option_price_close = get_option_price(options_data_today, current_position['Strike'], current_position['Option Type'], 'Close')
                        current_position['Exit Price'] = option_price_close if option_price_close is not None else correct_price_on_expiry(current_position['Strike'],spot,current_position['Option Type'])
                        current_position['Exit Delta'] = calculate_greeks(spot, current_position['Strike'], time_to_maturity, 0.07, volatility, current_position['Option Type'])
                        current_position['PNL']=(current_position['Initial Price']-current_position['Exit Price'])*current_position['Lot']
                        current_position['Exit Date']=current_date_str
                        option_trades.append(current_position.copy())
                        is_option_open=False
                    current_date+=timedelta(days=1)
                    continue

                valid_expriy_day=current_date
                spot=stock_data_today['EQ_Close'].values[0]
                volatility=stock_data_today['Volatility'].values[0]
                trend=stock_data_today['Trend'].values[0]
                
                if is_option_open==True:
                    now_trend='call'
                    if trend==1:
                        now_trend='put'
                    option_price_close = get_option_price(options_data_today, current_position['Strike'], current_position['Option Type'], 'Close')
                    if in_trade_option_type!=now_trend:
                        current_position['Exit Price'] = option_price_close if option_price_close is not None else current_position['Initial Price']
                        current_position['Exit Delta'] = calculate_greeks(spot, current_position['Strike'], time_to_maturity, 0.07, volatility, current_position['Option Type'])
                        current_position['PNL']=(current_position['Initial Price']-current_position['Exit Price'])*current_position['Lot']
                        current_position['Exit Date']=current_date_str
                        option_trades.append(current_position.copy())
                        is_option_open=False
                        option_target = find_option_by_delta(options_data_today, spot, time_to_maturity, volatility, 0.25 if trend == -1 else -0.25, 'call' if trend == -1 else 'put')
                        if option_target is not None:
                            option_lot_size=exposure/spot
                            current_position={
                                'Date':current_date_str,
                                'Entry':option_target['Ticker'],
                                'Strike':option_target['Strike Price'],
                                'Initial Price': option_target['Close'],
                                'Option Type':option_target['Extracted Option Type'],
                                'Bought Delta':option_target['Calculated_Delta'],
                                'Lot':option_lot_size,
                                'Expiry Month':lt.strftime('%Y-%m-%d'),
                                'Target Delta':0.25 
                            }
                            in_trade_option_type=option_target['Extracted Option Type']
                            is_option_open=True

                    if current_date==lt:
                        option_price_close = get_option_price(options_data_today, current_position['Strike'], current_position['Option Type'], 'Close')
                        current_position['Exit Price'] = option_price_close if option_price_close is not None else correct_price_on_expiry(current_position['Strike'],spot,current_position['Option Type'])
                        current_position['Exit Delta'] = calculate_greeks(spot, current_position['Strike'], time_to_maturity, 0.07, volatility, current_position['Option Type'])
                        current_position['PNL']=(current_position['Initial Price']-current_position['Exit Price'])*current_position['Lot']
                        current_position['Exit Date']=current_date_str
                        option_trades.append(current_position.copy())
                        is_option_open=False

                
                if is_option_open==False:
                    option_target = find_option_by_delta(options_data_today, spot, time_to_maturity, volatility, 0.25 if trend == -1 else -0.25, 'call' if trend == -1 else 'put')
                    if option_target is not None:
                        option_lot_size=exposure/spot
                        current_position={
                            'Date':current_date_str,
                            'Entry':option_target['Ticker'],
                            'Strike':option_target['Strike Price'],
                            'Initial Price': option_target['Close'],
                            'Option Type':option_target['Extracted Option Type'],
                            'Bought Delta':option_target['Calculated_Delta'],
                            'Lot':option_lot_size,
                            'Expiry Month':lt.strftime('%Y-%m-%d'),
                            'Target Delta':0.25 
                        }
                        in_trade_option_type=option_target['Extracted Option Type']
                        is_option_open=True

                current_date+=timedelta(days=1)
                

    return option_trades

# Define the function for multiprocessing
def get_data_multiprocessing(tickers, start_year, end_year, exposure):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(process_ticker, [(ticker, start_year, end_year, exposure) for ticker in tickers])

    # Flatten the list of lists
    all_trades = [trade for result in results for trade in result]
    return all_trades


if __name__ == '__main__':
    tickers = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "HINDUNILVR", "KOTAKBANK", "SBIN",
           "BHARTIARTL", "ITC", "ASIANPAINT", "BAJFINANCE", "MARUTI", "AXISBANK", "LT", "HCLTECH",
           "SUNPHARMA", "WIPRO", "ULTRACEMCO", "TITAN", "TECHM", "NESTLEIND", "JSWSTEEL", "TATASTEEL",
           "POWERGRID", "ONGC", "COALINDIA", "INDUSINDBK", "BAJAJFINSV", "GRASIM", "CIPLA", "ADANIPORTS",
           "TATAMOTORS", "DRREDDY", "BRITANNIA", "HEROMOTOCO", "DIVISLAB", "EICHERMOT", "SHREECEM",
           "APOLLOHOSP", "UPL", "TATACONSUM", "BAJAJ_AUTO", "HINDALCO", "SBILIFE", "VEDL"]
    start_year = 2019
    end_year = 2025
    exposure = 700000

    df_trades = get_data_multiprocessing(tickers, start_year, end_year, exposure)
    df = pd.DataFrame(df_trades)
    df.to_csv('trades.csv', index=False)
