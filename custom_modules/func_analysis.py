import emoji
import collections as c
import pandas as pd
import math
import streamlit as st
# for visualization
import plotly.express as px
import matplotlib.pyplot as plt
import re

# word cloud
from wordcloud import WordCloud, STOPWORDS


def authors_name(data):
    """
        It returns the name of participants in chat. 
    """
    authors = data.Author.unique().tolist()
    return [name for name in authors if name != None]


def extract_emojis(s):
    """
        This function is used to calculate emojis in text and return in a list.
    """
    return [c for c in s if c in emoji.EMOJI_DATA]


def stats(data):
    """
        This function takes input as data and return number of messages and total emojis used in chat.
    """
    total_messages = data.shape[0]
    media_messages = data[data['Message'] == '<Media omitted>'].shape[0]
    emojis = sum(data['emoji'].str.len())
    
    return "Total Messages 💬: {} \n Total Media 🎬: {} \n Total Emoji's 😂: {}".format(total_messages, media_messages, emojis)


def popular_emoji(data):
    """
        This function returns the list of emoji's with it's frequency.
    """
    total_emojis_list = list([a for b in data.emoji for a in b])
    emoji_dict = dict(c.Counter(total_emojis_list))
    emoji_list = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
    return emoji_list


def visualize_emoji(data):
    """
        This function is used to make pie chart of popular emoji's.
    """
    emoji_df = pd.DataFrame(popular_emoji(data), columns=['emoji', 'count'])
    
    fig = px.pie(emoji_df, values='count', names='emoji', color_discrete_map="identity", title='Emoji Distribution')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    # fig.show()
    return fig

def word_cloud(df):
    """
        This function is used to generate word cloud using dataframe.
    """
    df = df[df['Message'] != '<Media omitted>']
    df = df[df['Message'] != 'This message was deleted']
    words = ' '.join(df['Message'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
    # To stop article, punctuations
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=640, width=800).generate(processed_words)
    
    # plt.figure(figsize=(45,8))
    fig = plt.figure()
    ax = fig.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig
    

def active_date(data):
    """
    Visualizes the top 10 active dates based on the number of messages.

    Parameters:
    - data: DataFrame containing a 'Date' column.

    Returns:
    - fig: Matplotlib figure object.
    """
    # Count occurrences of each date and select the top 10
    top_dates = data['Date'].value_counts().head(10)

    # Create a horizontal bar plot
    fig, ax = plt.subplots()
    top_dates.plot.barh(ax=ax, color='Green', edgecolor='black')  # Customize colors if needed

    # Set plot title and axis labels
    ax.set_title('Top 10 Active Dates')
    ax.set_xlabel('Number of Messages')
    ax.set_ylabel('Date')

    # Display the count values on the bars
    for index, value in enumerate(top_dates):
        ax.text(value, index, str(value), ha='left', va='center', color='black')

    # Adjust layout for better appearance
    plt.tight_layout()

    return fig
    
def active_time(data):
    """
    This function generate horizontal bar graph between time and number of messages.

    Parameters
    ----------
    data : Dataframe
        With this data graph is generated.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots()
    ax = data['Time'].value_counts().head(10).plot.barh()
    ax.set_title('Top 10 active time')
    ax.set_xlabel('Number of messages')
    ax.set_ylabel('Time')
    plt.tight_layout()
    return fig

def day_wise_count(data):
    """
    This function generate a line polar plot.

    Parameters
    ----------
    data : DataFrame
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    day_df = pd.DataFrame(data["Message"])
    day_df['day_of_date'] = data['Date'].dt.weekday
    day_df['day_of_date'] = day_df["day_of_date"].apply(lambda d: days[d])
    day_df["messagecount"] = 1
    
    day = day_df.groupby("day_of_date").sum()
    day.reset_index(inplace=True)
    
    fig = px.line_polar(day, r='messagecount', theta='day_of_date', line_close=True)
    fig.update_traces(fill='toself')
    fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
        )),
    showlegend=False
    )
    # fig.show()
    return fig

def num_messages(data):
    """
    This function generates the line plot of number of messages on monthly basis.

    Parameters
    ----------
    data : DataFrame
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    data.loc[:, 'MessageCount'] = 1
    date_df = data.groupby("Date").sum()
    date_df.reset_index(inplace=True)
    fig = px.line(date_df, x="Date", y="MessageCount")
    fig.update_xaxes(nticks=20)
    # fig.show()
    return fig

def chatter(data):
    """
    This function generates a bar plot of members involve in a chat corressponding
    to the number of messages.

    Parameters
    ----------
    data : DataFrame
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    auth = data.groupby("Author").sum(numeric_only=True)
    
    auth.reset_index(inplace=True)
    fig = px.bar(auth, y="Author", x="MessageCount", color='Author', orientation="h",
             color_discrete_sequence=["red", "green", "blue", "goldenrod", "magenta"],
             title='Number of messages corresponding to author'
            )
    # fig.show()
    return fig

def user_percentage(data):
    total_messages = len(data)
    author_counts = data['Author'].value_counts()
    author_percentage = (author_counts / total_messages) * 100
    result = [(author, math.ceil(percentage)) for author, percentage in author_percentage.items()]
    emoji_df = pd.DataFrame(result, columns=['Author', 'percentage'])
    fig = px.pie(emoji_df, values='percentage', names='Author', color_discrete_map="identity", title='Percentage of Messages Sent by Each Author')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def user_with_longest_message(df):
    # Find the user with the longest message
    max_length_idx = df['Message'].apply(len).idxmax()
    user_with_longest_message = df.loc[max_length_idx, 'Author']
    longest_message = df.loc[max_length_idx, 'Message']
    return user_with_longest_message, longest_message
    
def messages_by_date(selected_date, df):
    selected_date = pd.to_datetime(selected_date, format='%Y-%m-%d', errors='coerce')
    selected_date_messages = df[df['Date'] == selected_date]

    if not selected_date_messages.empty:
        st.success(f"Messages on {selected_date}:")
        st.table(selected_date_messages[['Date', 'Time', 'Author', 'Message']])
    else:
        st.warning(f"No messages found on {selected_date}.")
















def display_longest_message(chat_data):
    # Find the index of the row with the longest message
    max_length_idx = chat_data['Message'].apply(len).idxmax()

    # Get information from the row with the longest message
    longest_message_author = chat_data.loc[max_length_idx, 'Author']
    longest_message_date = chat_data.loc[max_length_idx, 'Date']
    longest_message_time = chat_data.loc[max_length_idx, 'Time']
    longest_message_content = chat_data.loc[max_length_idx, 'Message']

    st.subheader("Longest Message:")
    st.write(f"Author: {longest_message_author}")
    st.write(f"Date: {longest_message_date}")
    st.write(f"Time: {longest_message_time}")
    st.write(f"Message: {longest_message_content}")


def yearly_comparison(data):
    
    # Extract the year from the 'Date' column
    data['Year'] = data['Date'].dt.year

    # Count the number of messages per year
    yearly_message_counts = data['Year'].value_counts().sort_index()

    # Create a DataFrame with years and their corresponding message counts
    yearly_message_df = pd.DataFrame({'Year': yearly_message_counts.index, 'MessageCount': yearly_message_counts.values})

    # Create a bar graph using Plotly
    fig = px.bar(yearly_message_df, x='Year', y='MessageCount', title='Yearly Comparison of Messages')
    fig.update_layout(xaxis_title='Year', yaxis_title='Number of Messages')

    return fig