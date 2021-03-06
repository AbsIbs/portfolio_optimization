from flask import Flask, render_template, request, jsonify
import yfinance as yf
from joblib import load
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
import plotly
import plotly.express as px
from plotly.io import to_json
import plotly.graph_objects as go
import json
import requests
from datetime import date
from decimal import Decimal

from finrl.apps import config_tickers
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.finrl_meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline,convert_daily_return_to_pyfolio_ts
from finrl.finrl_meta.data_processor import DataProcessor
from finrl.finrl_meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
from gym import spaces
from datetime import date

from ta.volatility import BollingerBands
from ta.trend import SMAIndicator, MACD, CCIIndicator, EMAIndicator, IchimokuIndicator, AroonIndicator
from ta.momentum import RSIIndicator, stoch, WilliamsRIndicator
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator 

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/callback/<endpoint>')
def cb(endpoint):   
    if endpoint == "getStock":
        return gm(request.args.get('data'),request.args.get('period'),request.args.get('interval'))
    else:
        return "Bad endpoint", 400

@app.route('/callback2/<endpoint>')
def cb2(endpoint):
    if endpoint == 'fetchROI':
        return retreive_roi()
    else:
        return "Bad endpoint", 400

@app.route('/callback3/<endpoint>')
def cb3(endpoint):
    if endpoint == 'backTest':
        return model(
            model_type='back_test', 
            start_date=request.args.get('startDate'), 
            end_date=request.args.get('endDate') 
            )
    elif endpoint == 'createModel':
        return model(
            model_type='create_model',
            start_date='start',
            end_date='end')
    else:
        return "Bad endpoint", 400

def gm(stock, period, interval):
    st = yf.Ticker(stock)
    data = st.history(period=period, interval=interval)
    st_info = st.info
    company_name = st_info['longName']
    company_sector = st_info['sector']
    company_ROI = round(((100 / data['Close'].iloc[0]) * data['Close'].iloc[-1]) - 100, 2)
    start_date = str(data.index[0])[:-9]
    end_date = str(data.index[-1])[:-9]
    data['ROI'] = data['Close'].pct_change()
    company_std = round(data['ROI'].dropna().std(), 2)

    def color(ROI):
        if ROI > 0:
            return 'green'
        else:
            return 'red'

    x = data.reset_index()['Date']
    y = data.reset_index()['Close']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y,
        fill='tozeroy',
        mode='lines',
        line_color=color(company_ROI),
        hoverinfo = 'x+y'
        ))
    fig.update_layout(title_text = f'Period: {start_date} to {end_date}')
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Closing Price ($)")
        
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    gm_dict = {
        'graphJson': graphJSON,
        'companyName': company_name,
        'companySector': company_sector,
        'companyROI': company_ROI,
        'companyStd': company_std
        }

    gm_dict = json.dumps(gm_dict)

    return gm_dict

def retreive_roi():
    tickers_list = [
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC", 
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "CRM",
    "VZ",
    "V",
    "WBA",
    "WMT",
    "DIS",
    "DOW"
    ]

    tickers = ' '.join(ticker for ticker in tickers_list)
    df = yf.download(tickers, period='5d')
    df = df.reset_index().melt('Date')
    df = df.rename(columns={'variable_0': 'values', 'variable_1': 'ticker'})
    df = df.pivot(index = ['Date', 'ticker'], columns = 'values', values='value').reset_index()

    roi_dict = {}
    for ticker in tickers_list:
        data = df[df['ticker'] == ticker]
        roi_current = data['Adj Close'].iloc[-1]
        roi_prior = data['Adj Close'].iloc[-2]
        roi = round(((100/roi_prior) * roi_current) - 100, 2)

        roi_dict[ticker] = roi

    sorted_roi_dict = {k: v for k, v in sorted(roi_dict.items(), key=lambda item: item[1], reverse=True)}

    sorted_roi_list = []
    for key, value in sorted_roi_dict.items():
        sorted_roi_list.append({'ticker': key, 'ROI': value})

    roi_df = pd.DataFrame(sorted_roi_list)
    DOW_sectors = pd.read_csv('static/models/DOW_sector.csv')
    DOW_roi_sectors_df = roi_df.merge(DOW_sectors)

    # Top risers
    fig_risers = px.bar(
    DOW_roi_sectors_df.iloc[0:3].sort_values('ROI'),
    x = 'ROI',
    y = 'ticker',
    orientation = 'h',
    color_discrete_sequence=["green"],
    hover_data = ['Sector'],
    title = 'Top Risers',
    height = 250
    )

    top_risersJSON = json.dumps(fig_risers, cls=plotly.utils.PlotlyJSONEncoder)

    # Top fallers
    fig_losers = px.bar(
    DOW_roi_sectors_df.iloc[-4:-1],
    x = 'ROI',
    y = 'ticker',
    orientation = 'h',
    color_discrete_sequence=["red"],
    hover_data = ['Sector'],
    title = 'Top Fallers',
    height = 250
    )

    fig_losers['layout']['xaxis']['autorange'] = "reversed"

    top_fallersJSON = json.dumps(fig_losers, cls=plotly.utils.PlotlyJSONEncoder)

    roi_dict = json.dumps(roi_dict)

    roi_summary_dict = {
        'top_risers': top_risersJSON,
        'top_fallers': top_fallersJSON,
        'roi_dict': roi_dict
    }
    
    return roi_summary_dict

def model(model_type, start_date, end_date):
    DOW_30_TICKERS = [
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC", 
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "CRM",
    "VZ",
    "V",
    "WBA",
    "WMT",
    "DIS",
    ]

    tickers = ' '.join(ticker for ticker in DOW_30_TICKERS)
    df = yf.download(tickers, period='3y')
    df = df.reset_index().melt('Date')
    df = df.rename(columns={'variable_0': 'values', 'variable_1': 'ticker'})
    df = df.pivot(index = ['Date', 'ticker'], columns = 'values', values='value').reset_index()
    df = df.dropna()
    df = pd.DataFrame(df.to_dict())

    # assign a column order for our dataframes
    column_order = ['date', 'open', 'high', 'low', 'close', 'adjcp', 'volume',	'tic', 'day']

    # make the index a datetime object
    df = df.rename({'Date': 'date', 'ticker': 'tic', 'Adj Close': 'adjcp'}, axis = 1)
    df['date'] = pd.to_datetime(df['date'])
    # add a column for 'day of the week'
    df['day'] = df['date'].dt.dayofweek
    # set a column order for our dataframes
    df.columns = df.columns.str.lower()
    df = df[column_order]
    # convert date column back to str
    df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    processed_df = FeatureEngineer(use_technical_indicator = True, use_turbulence = True).preprocess_data(df)

    #Log returns
    processed_df['log_return'] = np.log(processed_df['adjcp'] / processed_df['adjcp'].shift(1))
    # CCI
    _cci = CCIIndicator(high = processed_df['high'],
                        low = processed_df['low'], 
                        close = processed_df['close'],
                        window = 20)
    processed_df['20_day_CCI'] = _cci.cci()
    # Ichimoku Indicator
    ichi = IchimokuIndicator(high = processed_df['high'],
                                low = processed_df['low'])
                                
    processed_df['ichimoku_span_a'] = ichi.ichimoku_a()
    processed_df['ichimoku_span_b'] = ichi.ichimoku_b()
    processed_df['ichimoku_span_baseline'] = ichi.ichimoku_base_line()
    processed_df['ichimoku_span_conversion_line'] = ichi.ichimoku_conversion_line()
    # ArronIndicator
    _aroon = AroonIndicator(close = processed_df['close'])
    processed_df['Aroon_down'] = _aroon.aroon_down()
    processed_df['Aroon_up'] = _aroon.aroon_up()
    processed_df['Aroon_indicator'] = _aroon.aroon_indicator()
    #Williams R Indicator
    processed_df['wiilliams_r'] = WilliamsRIndicator(high = processed_df['high'],
                                            low = processed_df['low'],
                                            close = processed_df['close']
                                            ).williams_r()
    # On Balance Volume
    processed_df['on_balance_volume'] = OnBalanceVolumeIndicator(close = processed_df['close'],
                                                        volume = processed_df['volume']
                                                        ).on_balance_volume()
    #MFI
    processed_df['mfi'] = MFIIndicator(high = processed_df['high'],
                            low = processed_df['low'],
                            close = processed_df['close'],
                            volume = processed_df['volume']
                            ).money_flow_index()
                            
    DJIA = processed_df

    DJIA = DJIA.sort_values(['date', 'tic'], ignore_index=True)
    DJIA.index = DJIA.date.factorize()[0]

    cov_list = []
    return_list = []

    lookback = 252
    for i in range(lookback, len(DJIA.index.unique())):
        data_lookback = DJIA.loc[i - lookback: i, :]
        price_lookback = data_lookback.pivot_table(index = 'date', columns = 'tic', values = 'close')
        
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)

        covs = return_lookback.cov().values
        cov_list.append(covs)

    DJIA_cov = pd.DataFrame({'date': DJIA.date.unique()[lookback:], 'cov_list': cov_list, 'return_list': return_list})

    DJIA = DJIA.merge(DJIA_cov, on = 'date')
    DJIA = DJIA.sort_values(['date', 'tic']).reset_index(drop = True)

    class StockPortfolioEnv(gym.Env):
        """A single stock trading environment for OpenAI gym

        Attributes
        ----------
            df: DataFrame
                input data
            stock_dim : int
                number of unique stocks
            hmax : int
                maximum number of shares to trade
            initial_amount : int
                start money
            transaction_cost_pct: float
                transaction cost percentage per trade
            reward_scaling: float
                scaling factor for reward, good for training
            state_space: int
                the dimension of input features
            action_space: int
                equals stock dimension
            tech_indicator_list: list
                a list of technical indicator names
            turbulence_threshold: int
                a threshold to control risk aversion
            day: int
                an increment number to control date

        Methods
        -------
        _sell_stock()
            perform sell action based on the sign of the action
        _buy_stock()
            perform buy action based on the sign of the action
        step()
            at each step the agent will return actions, then 
            we will calculate the reward, and return the next observation.
        reset()
            reset the environment
        render()
            use render to return other functions
        save_asset_memory()
            return account value at each time step
        save_action_memory()
            return actions/positions at each time step
            

        """
        metadata = {'render.modes': ['human']}

        def __init__(self, 
                    df,
                    stock_dim,
                    hmax,
                    initial_amount,
                    transaction_cost_pct,
                    reward_scaling,
                    state_space,
                    action_space,
                    tech_indicator_list,
                    turbulence_threshold=None,
                    lookback=252,
                    day = 0):
            #super(StockEnv, self).__init__()
            #money = 10 , scope = 1
            self.day = day
            self.lookback=lookback
            self.df = df
            self.stock_dim = stock_dim
            self.hmax = hmax
            self.initial_amount = initial_amount
            self.transaction_cost_pct =transaction_cost_pct
            self.reward_scaling = reward_scaling
            self.state_space = state_space
            self.action_space = action_space
            self.tech_indicator_list = tech_indicator_list

            # action_space normalization and shape is self.stock_dim
            self.action_space = spaces.Box(low = 0, high = 1,shape = (self.action_space,)) 
            # Shape = (34, 30)
            # covariance matrix + technical indicators
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (self.state_space+len(self.tech_indicator_list),self.state_space))

            # load data from a pandas dataframe
            self.data = self.df.loc[self.day,:]
            self.covs = self.data['cov_list'].values[0]
            self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
            self.terminal = False     
            self.turbulence_threshold = turbulence_threshold        
            # initalize state: inital portfolio return + individual stock return + individual weights
            self.portfolio_value = self.initial_amount

            # memorize portfolio value each step
            self.asset_memory = [self.initial_amount]
            # memorize portfolio return each step
            self.portfolio_return_memory = [0]
            self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
            self.date_memory=[self.data.date.unique()[0]]

            
        def step(self, actions):
            # print(self.day)
            self.terminal = self.day >= len(self.df.index.unique())-1
            # print(actions)

            if self.terminal:
                df = pd.DataFrame(self.portfolio_return_memory)
                df.columns = ['daily_return']
                # plt.plot(df.daily_return.cumsum(),'r')
                # plt.savefig('static/results/cumulative_reward.png')
                # plt.close()
                
                # plt.plot(self.portfolio_return_memory,'r')
                # plt.savefig('static/results/rewards.png')
                # plt.close()

                print("=================================")
                print("begin_total_asset:{}".format(self.asset_memory[0]))           
                print("end_total_asset:{}".format(self.portfolio_value))

                df_daily_return = pd.DataFrame(self.portfolio_return_memory)
                df_daily_return.columns = ['daily_return']
                if df_daily_return['daily_return'].std() !=0:
                    sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/df_daily_return['daily_return'].std()
                    print("Sharpe: ",sharpe)
                print("=================================")
                
                return self.state, self.reward, self.terminal,{}

            else:
                #print("Model actions: ",actions)
                # actions are the portfolio weight
                # normalize to sum of 1
                #if (np.array(actions) - np.array(actions).min()).sum() != 0:
                #  norm_actions = (np.array(actions) - np.array(actions).min()) / (np.array(actions) - np.array(actions).min()).sum()
                #else:
                #  norm_actions = actions
                weights = self.softmax_normalization(actions) 
                #print("Normalized actions: ", weights)
                self.actions_memory.append(weights)
                last_day_memory = self.data

                #load next state
                self.day += 1
                self.data = self.df.loc[self.day,:]
                self.covs = self.data['cov_list'].values[0]
                self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
                #print(self.state)
                # calcualte portfolio return
                # individual stocks' return * weight
                portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)
                # update portfolio value
                new_portfolio_value = self.portfolio_value*(1+portfolio_return)
                self.portfolio_value = new_portfolio_value

                # save into memory
                self.portfolio_return_memory.append(portfolio_return)
                self.date_memory.append(self.data.date.unique()[0])            
                self.asset_memory.append(new_portfolio_value)

                # the reward is the new portfolio value or end portfolo value
                self.reward = new_portfolio_value 
                #print("Step reward: ", self.reward)
                #self.reward = self.reward*self.reward_scaling

            return self.state, self.reward, self.terminal, {}

        def reset(self):
            self.asset_memory = [self.initial_amount]
            self.day = 0
            self.data = self.df.loc[self.day,:]
            # load states
            self.covs = self.data['cov_list'].values[0]
            self.state =  np.append(np.array(self.covs), [self.data[tech].values.tolist() for tech in self.tech_indicator_list ], axis=0)
            self.portfolio_value = self.initial_amount
            #self.cost = 0
            #self.trades = 0
            self.terminal = False 
            self.portfolio_return_memory = [0]
            self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
            self.date_memory=[self.data.date.unique()[0]] 
            return self.state
        
        def render(self, mode='human'):
            return self.state
            
        def softmax_normalization(self, actions):
            numerator = np.exp(actions)
            denominator = np.sum(np.exp(actions))
            softmax_output = numerator/denominator
            return softmax_output

        
        def save_asset_memory(self):
            date_list = self.date_memory
            portfolio_return = self.portfolio_return_memory
            #print(len(date_list))
            #print(len(asset_list))
            df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
            return df_account_value

        def save_action_memory(self):
            # date and close price length must match actions length
            date_list = self.date_memory
            df_date = pd.DataFrame(date_list)
            df_date.columns = ['date']
            
            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            #df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
            return df_actions

        def _seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]

        def get_sb_env(self):
            e = DummyVecEnv([lambda: self])
            obs = e.reset()
            return e, obs

    if model_type == 'back_test':
        start_date = start_date
        end_date = end_date
        
    elif model_type == 'create_model':
        DJIA_dates = list(DJIA['date'])
        DJIA_dates = sorted(set(DJIA_dates), key=DJIA_dates.index)
        end_date = str(date.today())

        if date.today().weekday() < 4:
            start_date = DJIA_dates[-2]
        else: 
            start_date = DJIA_dates[-3]
            end_date = DJIA_dates[-1]
        
    
    test = data_split(DJIA, start_date, end_date)

    # observation space
    stock_dimension = len(test['tic'].unique())
    state_space = stock_dimension
    print(f'Stock Dimension: {stock_dimension}, State_space: {state_space}')

    technical_indicators_list = list(DJIA.columns[9:-2])

    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 1000000, 
        "transaction_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": technical_indicators_list, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4
    }
    e_trade_gym = StockPortfolioEnv(df = test, **env_kwargs)

    model = A2C.load('static/models/untuned_A2C')
    df_daily_return, df_actions = DRLAgent.DRL_prediction(
        model = model, 
        environment= e_trade_gym)

    if model_type == 'back_test':
        dow_index = yf.download('^DJI', period='5y')
        dow_index = dow_index.reset_index()
        dow_index = dow_index[(dow_index['Date'] > start_date) & (dow_index['Date'] < end_date)]
        dow_index['Date'] = dow_index['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        dow_index['ROI'] = dow_index['Adj Close'].pct_change()
        dow_index = dow_index.rename({'ROI': 'market_return', 'Date': 'date'}, axis = 1)
        combined_df = dow_index[['market_return', 'date']].merge(df_daily_return)
        combined_df = combined_df[['date', 'daily_return', 'market_return']]
        #combined_df['model_gain/loss'] = combined_df['daily_return'] - combined_df['market_return']

        combined_df['daily_return'] = combined_df['daily_return'].apply(lambda x: float(Decimal(x).quantize(Decimal('0.0001')) * 100))
        combined_df['market_return'] = combined_df['market_return'].apply(lambda x: float(Decimal(x).quantize(Decimal('0.0001')) * 100))
        combined_df['model_gain/loss'] = combined_df['daily_return'] - combined_df['market_return']
        combined_df['model_gain/loss'] = combined_df['model_gain/loss'].apply(lambda x: str(float(Decimal(x).quantize(Decimal('0.0001')))) + '%')

        fig = px.line(
            combined_df,
            x = 'date',
            y = ['daily_return', 'market_return'],
            hover_data= ['model_gain/loss']
        )
        fig.update_layout(title_text = f'Time Series')
        fig.update_layout({
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        })
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="ROI (%)")

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        market_vs_model = pd.DataFrame([
            {
                'Category': 'Model',
                'Cum_Sum': round(combined_df['daily_return'].sum(), 4),
            },
            {
                'Category': 'Market',
                'Cum_Sum': round(combined_df['market_return'].sum(), 4)
            }
        ])

        market_vs_model_fig = px.bar(
            market_vs_model,
            x = 'Category',
            y = 'Cum_Sum',
            color='Category'
        )

        market_vs_model_fig.update_layout(title_text = f'Model vs Market Cumulative ROI')
        market_vs_model_fig.update_yaxes(title_text="ROI (%)")
        market_vs_model_fig.update_layout({
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        })

        market_vs_model_graphJSON = json.dumps(market_vs_model_fig, cls=plotly.utils.PlotlyJSONEncoder)

        combined_df['daily_return'] = combined_df['daily_return'].apply(lambda x: str(x) + '%')
        combined_df['market_return'] = combined_df['market_return'].apply(lambda x: str(x) + '%')

        data_list = []

        for i in range(1, len(combined_df)):
            df = combined_df.iloc[i]
            data_list.append([
                df['date'],
                df['daily_return'],
                df['market_return'],
                df['model_gain/loss']
            ])

        data_result = {
            'graphJSON': graphJSON,
            'data': data_list,
            'market_vs_model': market_vs_model_graphJSON
        }

        data_result = json.dumps(data_result)

        return data_result


    elif model_type == 'create_model':
        new_df = df_actions.T
        new_df.columns = ['initial_holdings', 'recommended_holdings']
        new_df['initial_holdings'] = new_df['initial_holdings'].apply(lambda x: float(Decimal(x).quantize(Decimal('0.0001')) * 100))
        new_df['recommended_holdings'] = new_df['recommended_holdings'].apply(lambda x: float(Decimal(x).quantize(Decimal('0.0001')) * 100))
        new_df['holding_pct_change'] = new_df['recommended_holdings'] - new_df['initial_holdings']

        new_df['initial_holdings'] = new_df['initial_holdings'].apply(lambda x: str(x) + '%')
        new_df['recommended_holdings'] = new_df['recommended_holdings'].apply(lambda x: str(x) + '%')
        new_df['holding_pct_change'] = new_df['holding_pct_change'].apply(lambda x: str(float(Decimal(x).quantize(Decimal('0.0001')))) + '%')

        new_df = new_df.reset_index()
        new_df = new_df.rename({'index': 'ticker'}, axis = 1)

        data_list = []

        for ticker in new_df['ticker']:
            df = new_df[new_df['ticker'] == ticker]
            data_list.append([
                    list(df['ticker'])[0], 
                    list(df['initial_holdings'])[0], 
                    list(df['recommended_holdings'])[0], 
                    list(df['holding_pct_change'])[0]
                ])

        data_result = {
            'data': data_list
        }

        data_result = json.dumps(data_result)

        return data_result

    