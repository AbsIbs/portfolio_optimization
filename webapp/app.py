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
def chart1():
    greet = greeting('yo')
    make_plot(val_summary_DataFrame)
    return render_template('index.html', message = greet)

def greeting(text):
    return text

def make_plot(df):
    fig = px.bar(
        df,
        x = 'classifiers',
        y = 'val_accuracy',
        hover_data = ['val_recall', 'val_precision', 'val_f1'],
        color = 'val_accuracy',
        color_continuous_scale = px.colors.sequential.matter,
        title = 'NLP model summary',
        height = 720,
        width = 1280
    )

    fig.add_hline(
        y = 0.63,
        line_dash = 'dot',
        line_color = 'black'
    )

    fig.update_layout(
        yaxis = dict(
            title_text="Val Accuracy (%)"
        ),
        xaxis = dict(
            title_text = 'Classifiers'
        )
    )

    fig.show()