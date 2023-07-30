# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:58:35 2023

@author: user
"""
import preprocessor
import helper

# Importing modules
import nltk
import streamlit as st
import re
import preprocessor,helper
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud
stop_words_file = 'stop_hinglish.txt'
nltk.download('stopwords')
import nltk
nltk.download('vader_lexicon')

import subprocess

# Check if the 'wordcloud' package is installed
try:
    import wordcloud
except ImportError:
    # If 'wordcloud' package is not installed, install it
    subprocess.check_call(["pip", "install", "wordcloud"])

# Now you can import the 'wordcloud' package
import wordcloud

#Displaying the entire data frame

# App title
st.sidebar.title("Whatsapp Chat  Sentiment Analyzer")

# VADER : is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments.
nltk.download('vader_lexicon')

# File upload button
uploaded_file = st.sidebar.file_uploader("Choose a file")

# Main heading
st. markdown("<h1 style='text-align: center; color: grey;'>Whatsapp Chat Basic Analysis</h1>", unsafe_allow_html=True)


if uploaded_file is not None:
    
    # Getting byte form & then decoding
    bytes_data = uploaded_file.getvalue()
    d = bytes_data.decode("utf-8")
    # Perform preprocessing
    data = preprocessor.preprocess(d)
    df = preprocessor.preprocess(d)
    
    st. markdown("<h1 style='text-align: center; color: black;'></h1>", unsafe_allow_html=True)
    
    st. markdown("<h1 style='text-align: center; color: black;'>Overall Chat</h1>", unsafe_allow_html=True)
    st.dataframe(data)



    # Importing SentimentIntensityAnalyzer class from "nltk.sentiment.vader"
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    # Object
    sentiments = SentimentIntensityAnalyzer()
    
    # Creating different columns for (Positive/Negative/Neutral)
    data["po"] = [sentiments.polarity_scores(i)["pos"] for i in data["message"]] # Positive
    data["ne"] = [sentiments.polarity_scores(i)["neg"] for i in data["message"]] # Negative
    data["nu"] = [sentiments.polarity_scores(i)["neu"] for i in data["message"]] # Neutral
    
    # To indentify true sentiment per row in message column
    def sentiment(d):
        if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
            return 1
        if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
            return -1
        if d["nu"] >= d["po"] and d["nu"] >= d["ne"]:
            return 0

    # Creating new column & Applying function
    data['value'] = data.apply(lambda row: sentiment(row), axis=1)
    
    # User names list
    user_list = data['user'].unique().tolist()
    
    # Sorting
    user_list.sort()
    
    # Insert "Overall" at index 0
    user_list.insert(0, "Overall")
    
    # Selectbox
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)
 


            
    if st.sidebar.button("Show Analysis"):
        
#-------------------------------------------------
#Busy users
        num_messages,words,num_media_messages,num_links=helper.fetch_stats(selected_user,df)
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.header("Total Messages")
            
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)
            
        st.markdown("<h3 style='text-align: center; color: black;'>Most Busy Users",unsafe_allow_html=True)
        x,new_df = helper.most_busy_users(df)
        fig, ax = plt.subplots()
        col1, col2 = st.columns(2)
    
        with col1:
            bars = ax.bar(x.index, x.values,color='teal')
            for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
            f'{height}', ha='center', va='bottom')
            plt.xticks(rotation='vertical')
            ax.set_xlabel('User')  # Add x label
            ax.set_ylabel('Message Count')  # Add y label
            st.pyplot(fig)


        with col2:
            new_df_top_6 = new_df[:6]
            
            st.dataframe(new_df_top_6)
        
        col1, col2 = st.columns(2)
        
        #timeline
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline",unsafe_allow_html=True)
            timelinee = helper.monthly_timelinee(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timelinee['time'], timelinee['message'], color='navy')
            ax.set_xlabel('Month')  # Add x label
            ax.set_ylabel('Message Count')  # Add y label
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        

        #daily_timeline
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline",unsafe_allow_html=True)
            daily_timelinee = helper.daily_timelinee(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timelinee['only_date'], daily_timelinee['message'],color='magenta' )
            ax.set_xlabel('Date')  # Add x label
            ax.set_ylabel('Message Count')  # Add y label
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        
        
        col1,col2 = st.columns(2)


        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Most Busy Days</h3>",unsafe_allow_html=True)
            busy_day = helper.week_activity_map1(selected_user,df)
            fig,ax = plt.subplots()
            bars = ax.bar(busy_day.index,busy_day.values, color='pink')
            for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
            f'{height}', ha='center', va='bottom')
            plt.xticks(rotation='vertical')
            ax.set_xlabel('Days of the Week')  # Add a label to the x-axis

            ax.set_ylabel('Frequency')  # Add a label to the y-axis
            st.pyplot(fig)

        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Most Busy Months</h3>",unsafe_allow_html=True)
            busy_month = helper.month_activity_map1(selected_user, df)
            fig, ax = plt.subplots()
            bars = ax.bar(busy_month.index, busy_month.values, color='orange')
            for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
            f'{height}', ha='center', va='bottom')
            plt.xticks(rotation='vertical')
            ax.set_xlabel('Months')  # Add a label to the x-axis
            ax.set_ylabel('Frequency')  # Add a label to the y-axis
            st.pyplot(fig)
        
        
        col1,col2 = st.columns(2)
        
        with col1:
            #Most Common Words
            most_common_df = helper.most_common_words1(selected_user,df,stop_words_file)

            fig, ax = plt.subplots()
                
            ax.barh(most_common_df[0],most_common_df[1], color='cyan', height=0.5)
            plt.xticks(rotation='vertical')
            
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Words')

            st.markdown("<h3 style='text-align: center; color: black;'>Most Commonly Used Words</h3>",unsafe_allow_html=True)
            st.pyplot(fig)
        
        with col2:
            # Wordcloud
            st.markdown("<h3 style='text-align: center; color: black;'>Wordcloud</h3>", unsafe_allow_html=True)
            df_wc = helper.create_wordcloud1(selected_user, df,stop_words_file)
            fig, ax = plt.subplots(figsize=(8, 8))  # Set the figure size to match the WhatsApp logo dimensions
            ax.imshow(df_wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)


        #activity map

        st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Heatmap</h3>",unsafe_allow_html=True)
        user_heatmap1 = helper.activity_heatmap1(selected_user,df)
        fig,ax = plt.subplots()
        ax.set_xlabel("Weekday")
        ax.set_ylabel("Hour")
        ax = sns.heatmap(user_heatmap1)
        st.pyplot(fig)
        
       
#-----------------------------------------------------------------------------------   
        st. markdown("<h1 style='text-align: center; color: grey;'></h1>", unsafe_allow_html=True) 
        st. markdown("<h1 style='text-align: center; color: grey;'></h1>", unsafe_allow_html=True) 
        st. markdown("<h1 style='text-align: center; color: grey;'>Whatsapp Chat  Sentiment Analysis</h1>", unsafe_allow_html=True)
        st. markdown("<h1 style='text-align: center; color: grey;'></h1>", unsafe_allow_html=True)
        
        
        

        # Monthly activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Positive)</h3>",unsafe_allow_html=True)
            
            busy_month = helper.month_activity_map(selected_user, data,1)
            
            fig, ax = plt.subplots()
            bars = ax.bar(busy_month.index, busy_month.values, color='green')
            for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
            f'{height}', ha='center', va='bottom')
            ax.set_xlabel('Month')
            ax.set_ylabel('Activity Count (Positive)')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Neutral)</h3>",unsafe_allow_html=True)
            
            busy_month = helper.month_activity_map(selected_user, data, 0)
            
            fig, ax = plt.subplots()
            bars = ax.bar(busy_month.index, busy_month.values, color='grey')
            for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
            f'{height}', ha='center', va='bottom')
            ax.set_xlabel('Month')
            ax.set_ylabel('Activity Count (Neutral)')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Negative)</h3>",unsafe_allow_html=True)
            
            busy_month = helper.month_activity_map(selected_user, data, -1)
            
            fig, ax = plt.subplots()
            bars = ax.bar(busy_month.index, busy_month.values, color='red')
            for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
            f'{height}', ha='center', va='bottom')
            ax.set_xlabel('Month')
            ax.set_ylabel('Activity Count (Negative)')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Daily activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Positive)</h3>",unsafe_allow_html=True)
            
            busy_day = helper.week_activity_map(selected_user, data,1)
            
            fig, ax = plt.subplots()
            bars = ax.bar(busy_day.index, busy_day.values, color='green')
            for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
            f'{height}', ha='center', va='bottom')
            ax.set_xlabel('Day of Week')
            ax.set_ylabel('Activity Count (Positive)')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Neutral)</h3>",unsafe_allow_html=True)
            
            busy_day = helper.week_activity_map(selected_user, data, 0)
            
            fig, ax = plt.subplots()
            bars = ax.bar(busy_day.index, busy_day.values, color='grey')
            for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
            f'{height}', ha='center', va='bottom')
            ax.set_xlabel('Day of Week')
            ax.set_ylabel('Activity Count (Neutral)')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Negative)</h3>",unsafe_allow_html=True)
            
            busy_day = helper.week_activity_map(selected_user, data, -1)
            
            fig, ax = plt.subplots()
            bars = ax.bar(busy_day.index, busy_day.values, color='red')
            for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
            f'{height}', ha='center', va='bottom')
            ax.set_xlabel('Day of Week')
            ax.set_ylabel('Activity Count (Negative)')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Weekly activity map
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Positive)</h3>",unsafe_allow_html=True)
                
                user_heatmap = helper.activity_heatmap(selected_user, data, 1)
                
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                ax.set_xlabel('Day of Week')
                ax.set_ylabel('Activity Count (Positive)')
                st.pyplot(fig)
            except: 
                st.image('error.webp')
        with col2:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Neutral)</h3>",unsafe_allow_html=True)
                
                user_heatmap = helper.activity_heatmap(selected_user, data, 0)
                
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                ax.set_xlabel('Day of Week')
                ax.set_ylabel('Activity Count (Neutral)')
                st.pyplot(fig)
            except:
                st.image('error.webp')
        with col3:
            try:
                st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Negative)</h3>",unsafe_allow_html=True)
                
                user_heatmap = helper.activity_heatmap(selected_user, data, -1)
                
                fig, ax = plt.subplots()
                ax = sns.heatmap(user_heatmap)
                ax.set_xlabel('Day of Week')
                ax.set_ylabel('Activity Count (Negative)')
                st.pyplot(fig)
            except:
                st.image('error.webp')

        # Daily timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Positive)</h3>",unsafe_allow_html=True)
            
            daily_timeline = helper.daily_timeline(selected_user, data, 1)
            
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='green')
            ax.set_xlabel('Date')
            ax.set_ylabel('Activity Count (Positive)')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Neutral)</h3>",unsafe_allow_html=True)
            
            daily_timeline = helper.daily_timeline(selected_user, data, 0)
            
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='grey')
            ax.set_xlabel('Date')
            ax.set_ylabel('Activity Count (Neutral)')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Negative)</h3>",unsafe_allow_html=True)
            
            daily_timeline = helper.daily_timeline(selected_user, data, -1)
            
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='red')
            ax.set_xlabel('Date')
            ax.set_ylabel('Activity Count (Negative)')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Monthly timeline
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Positive)</h3>",unsafe_allow_html=True)
            
            timeline = helper.monthly_timeline(selected_user, data,1)
            
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='green')
            ax.set_xlabel('Month')
            ax.set_ylabel('Activity Count (Positive)')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Neutral)</h3>",unsafe_allow_html=True)
            
            timeline = helper.monthly_timeline(selected_user, data,0)
            
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='grey')
            ax.set_xlabel('Month')
            ax.set_ylabel('Activity Count (Neutral)')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col3:
            st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Negative)</h3>",unsafe_allow_html=True)
            
            timeline = helper.monthly_timeline(selected_user, data,-1)
            
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color='red')
            ax.set_xlabel('Month')
            ax.set_ylabel('Activity Count (Negative)')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Percentage contributed
        if selected_user == 'Overall':
            col1,col2,col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Contribution</h3>",unsafe_allow_html=True)
                x = helper.percentage(data, 1)
                
                # Displaying
                st.dataframe(x)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Contribution</h3>",unsafe_allow_html=True)
                y = helper.percentage(data, 0)
                
                # Displaying
                st.dataframe(y)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Contribution</h3>",unsafe_allow_html=True)
                z = helper.percentage(data, -1)
                
                # Displaying
                st.dataframe(z)


        # Most Positive,Negative,Neutral User...
        if selected_user == 'Overall':
            
            # Getting names per sentiment
            x = data['user'][data['value'] == 1].value_counts().head(10)
            y = data['user'][data['value'] == -1].value_counts().head(10)
            z = data['user'][data['value'] == 0].value_counts().head(10)

            col1,col2,col3 = st.columns(3)
            with col1:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Users</h3>",unsafe_allow_html=True)
                
                # Displaying
                fig, ax = plt.subplots()
                bars = ax.bar(x.index, x.values, color='green')
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
            f'{height}', ha='center', va='bottom')
                ax.set_xlabel('User(Positive)')
                ax.set_ylabel('Word_Count)')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Users</h3>",unsafe_allow_html=True)
                
                # Displaying
                fig, ax = plt.subplots()
                bars = ax.bar(z.index, z.values, color='grey')
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
            f'{height}', ha='center', va='bottom')
                ax.set_xlabel('User(Neutral)')
                ax.set_ylabel('Word_Count)')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Users</h3>",unsafe_allow_html=True)
                
                # Displaying
                fig, ax = plt.subplots()
                bars = ax.bar(y.index, y.values, color='red')
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
            f'{height}', ha='center', va='bottom')
                ax.set_xlabel('User(Negative)')
                ax.set_ylabel('Word_Count)')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

        # WORDCLOUD......
        col1,col2,col3 = st.columns(3)
        with col1:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Positive WordCloud</h3>",unsafe_allow_html=True)
                
                # Creating wordcloud of positive words
                df_wc = helper.create_wordcloud(selected_user, data,1)
                #fig, ax = plt.subplots()
                #ax.imshow(df_wc)
                #st.pyplot(fig)
                fig, ax = plt.subplots()
                ax.imshow(df_wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)                
            except:
                # Display error message
                st.image('error.webp')
        with col2:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Neutral WordCloud</h3>",unsafe_allow_html=True)
                
                # Creating wordcloud of neutral words
                df_wc = helper.create_wordcloud(selected_user, data,0)
                fig, ax = plt.subplots()
                ax.imshow(df_wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig) 
            except:
                # Display error message
                st.image('error.webp')
        with col3:
            try:
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Negative WordCloud</h3>",unsafe_allow_html=True)
                
                # Creating wordcloud of negative words
                df_wc = helper.create_wordcloud(selected_user, data,-1)
                fig, ax = plt.subplots()
                ax.imshow(df_wc, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig) 
            except:
                # Display error message
                st.image('error.webp')

        # Most common positive words
        col1, col2, col3 = st.columns(3)
        with col1:
            try:
                # Data frame of most common positive words.
                most_common_df = helper.most_common_words(selected_user, data,1)
                
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Positive Words</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1],color='green')
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Words (Positive)')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')
        with col2:
            try:
                # Data frame of most common neutral words.
                most_common_df = helper.most_common_words(selected_user, data,0)
                
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Neutral Words</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1],color='grey')
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Words (Neutral)')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')
        with col3:
            try:
                # Data frame of most common negative words.
                most_common_df = helper.most_common_words(selected_user, data,-1)
                
                # heading
                st.markdown("<h3 style='text-align: center; color: black;'>Negative Words</h3>",unsafe_allow_html=True)
                fig, ax = plt.subplots()
                ax.barh(most_common_df[0], most_common_df[1], color='red')
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Words (Negative)')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            except:
                # Disply error image
                st.image('error.webp')
                
                
                #-----------------------
        if selected_user == 'Overall': 
                    bytes_data = uploaded_file.getvalue()
                    data = bytes_data.decode("utf-8")
                    merged_df,avg_compound, avg_positive, avg_neutral, avg_negative = preprocessor.preprocess2(data)
                    st. markdown("<h1 style='text-align: center; color: grey;'>Chat Sentiments</h2>", unsafe_allow_html=True)
                    st.dataframe(merged_df)
                    
                    st.markdown(f"<h2 style='text-align: center;'>Average Compound Score: {avg_compound:.2f}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='text-align: center;'>Average Positive Score: {avg_positive:.2f}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='text-align: center;'>Average Neutral Score: {avg_neutral:.2f}</h2>", unsafe_allow_html=True)
                    st.markdown(f"<h2 style='text-align: center;'>Average Negative Score: {avg_negative:.2f}</h2>", unsafe_allow_html=True)
                #-----------------------

def main():
    
    # File Uploader in the sidebar
    uploaded_file = st.sidebar.file_uploader("Upload exported chat to check sentiment distribution", type=["txt"])
    
    if uploaded_file is not None:
        # Read the uploaded file as text
        data = uploaded_file.read().decode("utf-8")
        
        # Perform preprocessing and sentiment analysis
        _, avg_compound, avg_positive, avg_neutral, avg_negative = preprocessor.preprocess2(data)
        
        # Display pie chart of overall sentiment analysis
        show_sentiment_pie_chart(avg_positive, avg_neutral, avg_negative)

def show_sentiment_pie_chart(avg_positive, avg_neutral, avg_negative):
    # Create a DataFrame with the average scores
    average_scores = pd.DataFrame({
        'Sentiment': ['Positive', 'Neutral', 'Negative'],
        'Score': [avg_positive, avg_neutral, avg_negative]
    })
    
    # Increase the explosion value to explode the pie chart a little bit
    explode = (0.2, 0.2, 0.0)  # Tuple of three values corresponding to each slice

    st.markdown("<h3 style='text-align: center; color: black;'>Pie Chart for Overall Sentiment Distribution</h3>",unsafe_allow_html=True)
    # Plot the pie chart
    fig, ax = plt.subplots()
    ax.pie(average_scores['Score'], labels=average_scores['Sentiment'], autopct='%1.1f%%', startangle=90, explode=explode)
    ax.axis('equal')
    st.pyplot(fig)

if __name__ == "__main__":
    main()


