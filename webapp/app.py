from flask import Flask, render_template
from joblib import load
import numpy as numpy
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import plotly.express as px


app = Flask(__name__)
val_summary_DataFrame = pd.read_csv('data/NLP_summary_df.csv')

@app.route("/")
def index():
    return render_template('index.html')