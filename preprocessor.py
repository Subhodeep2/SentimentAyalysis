# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:04:24 2023

@author: user
"""

# Importing modules
import re
import pandas as pd
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
nltk.download('stopwords')
sia = SentimentIntensityAnalyzer()

# To convert text into data frame in desired form
def preprocess(data):
    
    # Regular expression
    #pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    
    # Split text file into messages & dates based on pattern
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    
    # Creating data frame
    df = pd.DataFrame({'user_message': messages, 'message_date': dates})
    
    #convert dates type
    try:
        df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %H:%M - ')
    except:
        df['message_date'] = pd.to_datetime(df['message_date'], format='%m/%d/%y, %H:%M - ')
       
   

        
    # Extract year from the dates
    df['year'] = df['message_date'].dt.strftime('%Y')  # Full year format
    df['year_short'] = df['message_date'].dt.strftime('%y')  # Short year format
    df.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df['user_message']:# For each message in user_message
        
        # Split message based on '([\w\W]+?):\s'
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]: 
            # User name
            users.append(entry[1])
            # Only message
            messages.append(" ".join(entry[2:]))
        else:
            # Adding group notifications
            users.append('group_notification')
            
            # Null value
            messages.append(entry[0])
    
    # Creating new columns
    df['user'] = users
    df['message'] = messages
    
    # Remove columns of no use
    df.drop(columns=['user_message'], inplace=True)
    
    # Extract date
    df['only_date'] = df['date'].dt.date
    
    # Extract year
    df['year'] = df['date'].dt.year
    
    # Extract month
    df['month_num'] = df['date'].dt.month
    
    # Extract month name
    df['month'] = df['date'].dt.month_name()
    
    # Extract day
    df['day'] = df['date'].dt.day
    
    # Extract day name
    df['day_name'] = df['date'].dt.day_name()
    
    # Extract hour
    df['hour'] = df['date'].dt.hour
    
    # Extract minute
    df['minute'] = df['date'].dt.minute

    # Remove entries having user as group_notification
    df = df[df['user'] != 'group_notification']
    
    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period
    
    
   # df_sa=df.drop(['date', 'only_date', 'year', 'month_num', 'day', 'day_name', 'hour', 'minute', 'period'], axis=1)
    # Returning preprocessed data frame
    return df

#----------------------------------------------------------------------------------------
def preprocess2(data):
    
    # Regular expression
    pattern = '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s'
    
    # Split text file into messages & dates based on pattern
    messages = re.split(pattern, data)[1:]
    dates = re.findall(pattern, data)
    
    # Creating data frame
    df_sa = pd.DataFrame({'user_message': messages, 'message_date': dates})
    
    # convert dates type
    try:
        df_sa['message_date'] = pd.to_datetime(df_sa['message_date'], format='%d/%m/%y, %H:%M - ')
    except:
        df_sa['message_date'] = pd.to_datetime(df_sa['message_date'], format='%m/%d/%y, %H:%M - ')
    df_sa.rename(columns={'message_date': 'date'}, inplace=True)

    users = []
    messages = []
    for message in df_sa['user_message']:# For each message in user_message
        
        # Split message based on '([\w\W]+?):\s'
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]: 
            # User name
            users.append(entry[1])
            # Only message
            messages.append(" ".join(entry[2:]))
        else:
            # Adding group notifications
            users.append('group_notification')
            
            # Null value
            messages.append(entry[0])
    
    # Creating new columns
    df_sa['user'] = users
    df_sa['message'] = messages
    
    # Remove columns of no use
    df_sa.drop(columns=['user_message'], inplace=True)
    df_sa.drop(columns=['date'], inplace=True)
    df_sa = df_sa.assign(ID=range(1, len(df_sa)+1))
    df_sa = df_sa.reindex(columns=['ID', 'user', 'message'])
    
    # Run the polarity score on the entire dataset
    res = {}
    for i, row in tqdm(df_sa.iterrows(), total=len(df_sa)):
        text = row['message']
        myid = row['ID']
        res[myid] = sia.polarity_scores(text)
        
    vaders = pd.DataFrame(res).T
    vaders = vaders.reset_index().rename(columns={'index': 'ID'})
    
    merged_df=pd.merge(df_sa,vaders, on='ID')
    
    # Calculate the average compound score
    avg_compound = merged_df['compound'].mean()
    
    # Calculate the average positive, neutral, and negative scores
    avg_positive = merged_df['pos'].mean()
    avg_neutral = merged_df['neu'].mean()
    avg_negative = merged_df['neg'].mean()

    return merged_df, avg_compound, avg_positive, avg_neutral, avg_negative
    


    
