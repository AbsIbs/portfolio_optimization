from flask import Flask, render_template, request
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

@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/callback/<endpoint>')
def cb(endpoint):   
    if endpoint == "getStock":
        return gm(request.args.get('data'),request.args.get('period'),request.args.get('interval'))
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