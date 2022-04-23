from flask import Flask, render_template, request, jsonify
import yfinance as yf
from joblib import load
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import plotly
import plotly.express as px
from plotly.io import to_json
import plotly.graph_objects as go
import json
import requests

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

def text_preprocessing(text):
    # lowercase
    text = text.lower()
    #remove ponctuation
    cleaned_text = re.sub("([^A-Za-z0-9|\s|[:punct:]]*)", '', text)
    # tokenize
    text_list = cleaned_text.split()
    # remove stopwords
    tokens = [word for word in text_list if word not in complete_stopwords_list]
    # lemmatize
    lemmatize = WordNetLemmatizer()
    lemmatized_list = [lemmatize.lemmatize(word) for word in tokens if len(lemmatize.lemmatize(word)) > 2]
    # rejoin our tokens into sentences
    words = ' '.join([word for word in lemmatized_list])
    return words

def vectorization(preprocessed_text):
    with open('NLP_models/count_vectorizer.pickle', 'rb') as handle:
        cv = pickle.load(handle)

    vectorized_text = cv.transform(preprocessed_text)
    return vectorized_text

def NLP_prediciton(vectorized_text):
    with open('NLP_models/tuned_gradient_boosting_model.pickle', 'rb') as handle:
        nlp_model = pickle.load(handle)
    
    prediction = nlp_model.predict(vectorized_text)
    return prediction
