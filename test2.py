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


def addtransaction(entry,exit,type):
    if entry==exit:
        return 0
    if type=='buy':
        n_exit=exit * 0.99
        n_entry=entry * 1.01
        return (n_exit-n_entry)
    else:
        n_entry=entry * 0.99 
        n_exit=exit * 1.01
        return (n_entry-n_exit)


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

def calculate_historical_volatility_nifty(equity_data, lookback_period=252):
    log_returns = np.log(equity_data['Close'] / equity_data['Close'].shift(1))
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
        return max(strike-spot,0)
    

def give_pnl(type,initial,exit,lot):
    if type=='BUY':
        initial=initial+(initial*0.01)
        exit=exit-(exit*0.01)
        return (exit-initial)*lot
    else :
        initial=initial-(initial*0.01)
        exit=exit+(exit*0.01)
        return (initial-exit)*lot

import multiprocessing
import pandas as pd
from datetime import timedelta


# Define the function to process each ticker in parallel
def process_ticker(ticker, start_year, end_year, exposure,target_delta,dte,sl,option_type):
    option_trades = []
    stock_data = pd.read_csv(f"./Stocks Data/{ticker}_EQ_EOD.csv")
    nifty_index_data = pd.read_csv('./nifty_combined_sorted_data.csv')

    #volatility
    result =calculate_historical_volatility(stock_data)
    stock_data['Volatility'] = result['volatility']
    result= calculate_historical_volatility_nifty(nifty_index_data)
    nifty_index_data['Volatility'] = result['volatility']

    #convert date to common format
    stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.strftime('%Y-%m-%d')
    nifty_index_data['Date'] = pd.to_datetime(nifty_index_data['Date']).dt.strftime('%Y-%m-%d')

    #options data
    op_data=pd.read_csv(f"./Stocks Data/{ticker}_Opt_EOD.csv")
    op_data['Date'] = pd.to_datetime(op_data['Date']).dt.strftime('%Y-%m-%d')
    op_data[['Strike Price', 'Extracted Option Type']] = op_data['Ticker'].apply(extract_strike_price_and_type).apply(pd.Series)

    #nifty options data
    nifty_options_data = pd.read_csv('./Nifty_MonthlyI_Opt2019.csv')
    nifty_options_data['Date'] = pd.to_datetime(nifty_options_data['Date']).dt.strftime('%Y-%m-%d')
    nifty_options_data[['Strike Price', 'Extracted Option Type']] = nifty_options_data['Ticker'].apply(extract_strike_price_and_type).apply(pd.Series)



    
    for year in range(start_year, end_year):
        for month in range(1, 13):
            #last friday
            lf=last_friday_of_previous_month(year, month)
            #last thursday
            lt = last_thursday(year, month)

            lf_str = lf.strftime('%Y-%m-%d')
            lt_str = lt.strftime('%Y-%m-%d')


            # filter stock and options data in range
            filter_stock_data = stock_data[(stock_data['Date'] >= lf_str) & (stock_data['Date'] <= lt_str)]
            filter_op_data = op_data[(op_data['Date'] >= lf_str) & (op_data['Date'] <= lt_str)]
            filter_nifty_op_data = nifty_options_data[(nifty_options_data['Date'] >= lf_str) & (nifty_options_data['Date'] <= lt_str)]
            filter_nifty_index_data = nifty_index_data[(nifty_index_data['Date'] >= lf_str) & (nifty_index_data['Date'] <= lt_str)]



            current_date = lf
            
            is_option_open=False
            is_nifty_option_open=False
            valid_expriy_day=None 
            
            while current_date <= lt:

                current_date_str=current_date.strftime('%Y-%m-%d')
                stock_data_today=filter_stock_data[filter_stock_data['Date']==current_date_str]
                options_data_today=filter_op_data[filter_op_data['Date']==current_date_str]
                nifty_op_data_today=filter_nifty_op_data[filter_nifty_op_data['Date']==current_date_str]
                nifty_index_data_today=filter_nifty_index_data[filter_nifty_index_data['Date']==current_date_str]
                time_to_maturity = calculate_time_to_maturity(current_date,lt)


                if len(stock_data_today)==0 or len(options_data_today)==0:
                    if current_date==valid_expriy_day:
                        valid_expriy_day_str=valid_expriy_day.strftime('%Y-%m-%d')
                        stock_data_today=filter_stock_data[filter_stock_data['Date']==valid_expriy_day_str]
                        options_data_today=filter_op_data[filter_op_data['Date']==valid_expriy_day_str]
                        # nifty_op_data_today=filter_nifty_op_data[filter_nifty_op_data['Date']==valid_expriy_day_str]
                        # nifty_index_data_today=filter_nifty_index_data[filter_nifty_index_data['Date']==valid_expriy_day_str]
                        time_to_maturity = calculate_time_to_maturity(valid_expriy_day,lt)
                        spot=stock_data_today['EQ_Close'].values[0]
                        volatility=stock_data_today['Volatility'].values[0]
                        option_price_close = get_option_price(options_data_today, current_position['Option Strike'], option_type, 'Close')
                        option_exit_price = option_price_close
                        if option_exit_price==None:
                            option_exit_price=correct_price_on_expiry(current_position['Option Strike'],spot,current_position['Option Type'])
                        current_position['Options PNL'] = give_pnl('SELL',current_position['Option Initial Price'],option_exit_price, current_position['lot_size'])
                        current_position['Option Close Date'] = current_date
                        current_position['Option Final Price'] = option_exit_price
                        current_position['Option SL'] = 'Expiry'
                        option_trades.append(current_position.copy())
                        is_option_open = False



                    current_date+=timedelta(days=1)
                    continue
                    

                valid_expriy_day=current_date
                spot=stock_data_today['EQ_Close'].values[0]
                spot_nifty=nifty_index_data_today['Close'].values[0]
                volatility=stock_data_today['Volatility'].values[0]
                nifty_volatility=nifty_index_data_today['Volatility'].values[0]


                if is_option_open==True:
                    print(current_position)
                    option_price_high = get_option_price(options_data_today, current_position['Option Strike'], option_type, 'High')
                    option_price_open = get_option_price(options_data_today, current_position['Option Strike'], option_type, 'Open')
                    option_price_close= get_option_price(options_data_today, current_position['Option Strike'], option_type, 'Close')
                    
                    if (option_price_open!=None) and (is_option_open==True):
                        if option_price_open >= (1 + sl) * current_position['Option Initial Price']:
                            #sell at open
                            option_exit_price = option_price_open
                            print("overnight sl hit"," ",option_exit_price)
                            current_position['Options PNL'] = give_pnl('SELL',current_position['Option Initial Price'],option_exit_price, current_position['lot_size'])  # Selling 2 lots
                            current_position['Option Close Date'] = current_date
                            current_position['Option Final Price'] = option_exit_price
                            current_position['Option SL'] = 'Overnight SL Hit'
                            option_trades.append(current_position.copy())
                            is_option_open = False

                    if (option_price_high!=None) and (is_option_open==True):
                        if option_price_high >= (1 + sl) * current_position['Option Initial Price']:
                            #sell at stoploss
                            option_exit_price = (1 + sl) * current_position['Option Initial Price']
                            print("intraday sl hit"," ",option_exit_price)
                            current_position['Options PNL'] = give_pnl('SELL',current_position['Option Initial Price'],option_exit_price, current_position['lot_size'])
                            current_position['Option Close Date'] = current_date
                            current_position['Option Final Price'] = option_exit_price
                            current_position['Option SL'] = 'Intraday SL Hit'
                            option_trades.append(current_position.copy())
                            is_option_open = False
                    
                    if (current_date==lt) and (is_option_open==True) :
                        #sell at expiry
                        option_exit_price = option_price_close
                        print("expiry sl hit"," ",option_exit_price)
                        if option_exit_price==None:
                            option_exit_price=correct_price_on_expiry(current_position['Option Strike'],spot,current_position['Option Type'])
                        current_position['Options PNL'] = give_pnl('SELL',current_position['Option Initial Price'],option_exit_price, current_position['lot_size'])
                        current_position['Option Close Date'] = current_date
                        current_position['Option Final Price'] = option_exit_price
                        current_position['Option SL'] = 'Expiry'
                        option_trades.append(current_position.copy())
                        is_option_open = False


                

            

                    
                       

                if is_option_open==False :
                    #stock option sell
                    option_target = find_option_by_delta(options_data_today, spot, time_to_maturity, volatility, target_delta, option_type)
                    

                    if option_target is not None:
                        #Month Start Enter AT Open
                        option_lot_size=exposure/spot
                        current_position={
                            'Date':current_date_str,
                            'Entry':option_target['Ticker'],
                            'Option Strike':option_target['Strike Price'],
                            'Option Initial Price': option_target['Close'],
                            'Option Type':option_target['Extracted Option Type'],
                            'Bought Delta':option_target['Calculated_Delta'],
                            'lot_size':option_lot_size,
                            'Expiry Month':lt.strftime('%Y-%m-%d'),
                            'Target Delta':target_delta,
                        }
                        in_trade_option_type=option_target['Extracted Option Type']
                        is_option_open=True
                        print(current_position)

                current_date+=timedelta(days=1)
                

    return option_trades

# Define the function for multiprocessing
def get_data_multiprocessing(tickers, start_year, end_year, exposure,target_delta,dte,sl,option_type):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(process_ticker, [(ticker, start_year, end_year, exposure,target_delta,dte,sl,option_type) for ticker in tickers])

    # Flatten the list of lists
    all_trades = [trade for result in results for trade in result]
    return all_trades


if __name__ == '__main__':
    tickers = ["RELIANCE"]
    start_year = 2019
    end_year = 2025
    exposure = 700000
    target_delta=0.35
    dte=5
    sl=2
    option_type='call'

    df_trades = get_data_multiprocessing(tickers, start_year, end_year, exposure,target_delta,dte,sl,option_type)
    df = pd.DataFrame(df_trades)
    df.to_csv(f'stocks_trades_{target_delta}_{dte}_{sl}_{option_type}.csv', index=False)
