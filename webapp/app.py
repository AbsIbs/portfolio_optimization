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
import json
import requests

app = Flask(__name__)

@app.before_first_request
def DOW_df():
    


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
    data = yf.download(stock, period=period, interval=interval)
    company_name = yf.Ticker(stock).info['longName']
    fig = px.area(
        data.reset_index(),
        x = 'Date',
        y = 'Adj Close',
        hover_data = ['Open', 'High', 'Low', 'Volume'],
        title = company_name
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON