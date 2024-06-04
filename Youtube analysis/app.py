import os
import nltk
import requests
import numpy as np
from bs4 import BeautifulSoup
import urllib.request as urllib
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from flask import Flask, render_template, request
import time

from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.service import Service

nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

wnl = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = stopwords.words('english')

def return_yt_comments(url):
    data = []

    # Specify the path to the chromedriver.exe file
    chromedriver_path = r'C:\Users\reliance\Desktop\chromedriver-win64\chromedriver.exe'
    service = Service(chromedriver_path)

    with Chrome(service=service) as driver:
        wait = WebDriverWait(driver, 15)
        driver.get(url)

        for item in range(5):
            wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
            time.sleep(2)

        for comment in wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content"))):
            data.append(comment.text)

    return data

def clean(org_comments):
    cleaned_comments = []
    for comment in org_comments:
        words = comment.split()
        words = [word.lower().strip() for word in words]
        words = [word for word in words if word not in stop_words]
        words = [word for word in words if len(word) > 2]
        words = [wnl.lemmatize(word) for word in words]
        cleaned_comments.append(' '.join(words))
    return cleaned_comments

def create_wordcloud(clean_reviews):
    if not clean_reviews:
        print("No comments to generate word cloud.")
        return

    for_wc = ' '.join(clean_reviews)
    wcstops = set(STOPWORDS)
    wc = WordCloud(width=1400, height=800, stopwords=wcstops, background_color='white').generate(for_wc)
    plt.figure(figsize=(20, 10), facecolor='k', edgecolor='k')
    plt.imshow(wc, interpolation='bicubic')
    plt.axis('off')
    plt.tight_layout()
    clean_cache(directory='static/images')
    plt.savefig('static/images/woc.png')
    plt.close()

def return_sentiment(x):
    score = sia.polarity_scores(x)['compound']

    if score > 0:
        sent = 'Positive'
    elif score == 0:
        sent = 'Negative'
    else:
        sent = 'Neutral'
    return score, sent

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/results', methods=['GET'])
def result():
    url = request.args.get('url')

    org_comments = return_yt_comments(url)
    temp_comments = [comment for comment in org_comments if 5 < len(comment) <= 500]
    clean_comments = clean(temp_comments)

    create_wordcloud(clean_comments)

    np, nn, nne = 0, 0, 0
    predictions = []
    scores = []

    for comment in clean_comments:
        score, sent = return_sentiment(comment)
        scores.append(score)
        if sent == 'Positive':
            predictions.append('POSITIVE')
            np += 1
        elif sent == 'Negative':
            predictions.append('NEGATIVE')
            nn += 1
        else:
            predictions.append('NEUTRAL')
            nne += 1

    result_dict = [{'sent': pred, 'clean_comment': cc, 'org_comment': oc, 'score': sc}
                   for pred, cc, oc, sc in zip(predictions, clean_comments, temp_comments, scores)]

    return render_template('result.html', n=len(clean_comments), nn=nn, np=np, nne=nne, dic=result_dict)

@app.route('/wc')
def wc():
    return render_template('wc.html')

def clean_cache(directory=None):
    '''
    This function is responsible for clearing any residual csv and image files
    present due to the past searches made.
    '''
    if directory is not None:
        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        if os.listdir(directory):
            # Iterate over the files and remove each file
            files = os.listdir(directory)
            for file_name in files:
                print(file_name)
                os.remove(os.path.join(directory, file_name))
        print("Cleaned!")
    else:
        print("Directory not provided.")


if __name__ == '__main__':
    app.run(debug=True)
