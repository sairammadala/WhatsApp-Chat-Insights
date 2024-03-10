from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import pickle
import nltk.classify.util
import plotly.express as px
import pandas as pd

def clean(words):
    return dict([(word, True) for word in words])

def sentiments(contents):
    f = open('model', 'rb')
    classifier = pickle.load(f)
    f.close()
    opinion = {}
    pos, neg = 0, 0
    for line in contents:
        try:
            chat = line.split('-')[1].split(':')[1]
            name = line.split('-')[1].split(':')[0]
            if opinion.get(name, None) is None:
                opinion[name] = [0, 0]
            res = classifier.classify(clean(chat))
            if res == 'positive':
                pos += 1
                opinion[name][0] += 1
            else:
                neg += 1
                opinion[name][1] += 1
        except:
            pass
    print("positive: {} \nNegative: {}".format(pos, neg))
    return opinion, pos, neg

def pie_chart(pos, neg):
    neg = abs(neg) if neg != 0 else 0
    neg = abs(neg)
    labels = ['Positive', 'Negative']
    sizes = [pos, neg]

    df = pd.DataFrame({'Sentiment': labels, 'Count': sizes})

    fig = px.pie(df, names='Sentiment', values='Count', title='WhatsApp Sentiment Analysis',
                 color_discrete_map={'Positive': 'lightgreen', 'Negative': 'lightcoral'})

    return fig

def bar_plot(opinion):
    names, positive, negative = [], [], []
    for name in opinion:
        names.append(name)
        positive.append(opinion[name][0])
        negative.append(opinion[name][1])

    df = pd.DataFrame({'Names': names, 'Positive': positive, 'Negative': negative})

    fig = px.bar(df, x='Names', y=['Positive', 'Negative'], labels={'value': 'Sentiment'},
                 color_discrete_map={'Positive': 'lightgreen', 'Negative': 'lightcoral'},
                 title='WhatsApp Chat Sentiment Analysis')



    return fig


def autolabel(rects, ax):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')
        


def top_negative(opinion):
    names, positive, negative = [], [], []
    for name in opinion:
        names.append(name)
        positive.append(opinion[name][0])
        negative.append(opinion[name][1])

    # Get top 5 positive and top 5 negative users
    top_pos_users = [name for _, name in sorted(zip(positive, names), reverse=True)[:5]]
    top_neg_users = [name for _, name in sorted(zip(negative, names), reverse=True)[:5]]

    print("Top 5 Positive Users:", top_pos_users)
    print("Top 5 Negative Users:", top_neg_users)

    return top_neg_users



def top_pos(opinion):
    names, positive, negative = [], [], []
    for name in opinion:
        names.append(name)
        positive.append(opinion[name][0])
        negative.append(opinion[name][1])

    

    # Get top 5 positive and top 5 negative users
    top_pos_users = [name for _, name in sorted(zip(positive, names), reverse=True)[:5]]
    top_neg_users = [name for _, name in sorted(zip(negative, names), reverse=True)[:5]]

    print("Top 5 Positive Users:", top_pos_users)
    print("Top 5 Negative Users:", top_neg_users)

    return top_pos_users