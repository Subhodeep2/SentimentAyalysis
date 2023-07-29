# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:06:07 2023

@author: user
"""

# Importing modules
import streamlit as st
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import numpy as np
from PIL import Image
stop_words_file = 'stop_hinglish.txt' 
whatsapp_logo = np.array(Image.open("whatsapp_logo.png"))
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
# Object
extract = URLExtract()

import subprocess

# Check if the 'wordcloud' package is installed
try:
    import wordcloud
except ImportError:
    # If 'wordcloud' package is not installed, install it
    subprocess.check_call(["pip", "install", "wordcloud"])

# Now you can import the 'wordcloud' package
import wordcloud

# -1 => Negative
# 0 => Neutral
# 1 => Positive

#------------------------------------

def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
        # 1. fetch number of messages
    num_messages = df.shape[0]
    # 2. fetch the number of words
    words = []
    for message in df['message']:
        words.extend(message.split())
    # fetch number of media messages
    num_media_messages = df[df['message']=='<Media omitted>\n'].shape[0]
    #fetch number of links shared
    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))
    return num_messages,len(words),num_media_messages,len(links)


def most_busy_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={'index': 'name', 'user': 'percent'})
    return x,df
    
def monthly_timelinee(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timelinee = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    time = []
    for i in range(timelinee.shape[0]):
        time.append(timelinee['month'][i] + '-' + str(timelinee['year'][i]))

    timelinee['time'] = time
    return timelinee

def daily_timelinee(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    daily_timelinee = df.groupby('only_date').count()['message'].reset_index()

    return daily_timelinee 

def week_activity_map1(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map1(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def load_stopwords(file_path):
    with open('stop_hinglish.txt', 'r') as file:
        stopwords = [line.strip() for line in file]
    return stopwords

#def most_common_words1(selected_user,df):

def most_common_words1(selected_user, df, stop_words_file):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[~df['message'].str.contains('<media', case=False, na=False)]

    stop_words = load_stopwords(stop_words_file)

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    return_df = pd.DataFrame(Counter(words).most_common(20))
    return return_df


def create_wordcloud1(selected_user,df, stop_words_file):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Remove '<Media omitted>' from the messages
    df['message'] = df['message'].replace('<Media omitted>', '', regex=True)

    stop_words = load_stopwords(stop_words_file)

    text = ' '.join(df['message'].str.lower())

    wordcloud = WordCloud(
        width=500,
        height=500,
        min_font_size=10,
        background_color='white',
        stopwords=stop_words,
        mask=whatsapp_logo,  # Set the word cloud shape to the WhatsApp logo
        contour_color='black',  # Optional: add a black contour around the logo
    )

    wordcloud.generate(text)

    return wordcloud


def activity_heatmap1(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap1 = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap1


#------------------------------------
# Will return count of messages of selected user per day having k(0/1/-1) sentiment
def week_activity_map(selected_user,df,k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    return df['day_name'].value_counts()


# Will return count of messages of selected user per month having k(0/1/-1) sentiment
def month_activity_map(selected_user,df,k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    return df['month'].value_counts()

# Will return heat map containing count of messages having k(0/1/-1) sentiment
def activity_heatmap(selected_user,df,k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value'] == k]
    
    # Creating heat map
    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    return user_heatmap


# Will return count of messages of selected user per date having k(0/1/-1) sentiment
def daily_timeline(selected_user,df,k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value']==k]
    # count of message on a specific date
    daily_timeline = df.groupby('only_date').count()['message'].reset_index()
    return daily_timeline


# Will return count of messages of selected user per {year + month number + month} having k(0/1/-1) sentiment
def monthly_timeline(selected_user,df,k):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    df = df[df['value']==-k]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))
    timeline['time'] = time
    return timeline

# Will return percentage of message contributed having k(0/1/-1) sentiment
def percentage(df,k):
    df = round((df['user'][df['value']==k].value_counts() / df[df['value']==k].shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return df

# Return wordcloud from words in message
def create_wordcloud(selected_user,df,k):
    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    
    # Remove entries of no significance
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    
    # Remove stop words according to text file "stop_hinglish.txt"
    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)
    # Dimensions of wordcloud
    #wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    # Dimensions of word cloud with WhatsApp shape
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white', mask=whatsapp_logo)
    
    # Actual removing
    temp['message'] = temp['message'].apply(remove_stop_words)
    temp['message'] = temp['message'][temp['value'] == k]
    
    # Word cloud generated
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

# Return set of most common words having k(0/1/-1) sentiment
def most_common_words(selected_user,df,k):
    f = open('stop_hinglish.txt','r')
    stop_words = f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']
    words = []
    for message in temp['message'][temp['value'] == k]:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
                
    # Creating data frame of most common 20 entries
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

