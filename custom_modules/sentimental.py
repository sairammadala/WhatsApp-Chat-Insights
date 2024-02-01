from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import pickle
import nltk.classify.util
import sys
import matplotlib.pyplot as plt
import numpy as np


def clean(words):
    return dict([(word, True) for word in words])

def sentiments(contents):
    f = open('model', 'rb')
    classifier = pickle.load(f)
    f.close()
    opinion={}
    pos,neg=0,0
    for line in contents:
        try:
            chat=line.split('-')[1].split(':')[1]
            name=line.split('-')[1].split(':')[0]
            if opinion.get(name,None) is None:
                opinion[name]=[0,0]
            res=classifier.classify(clean(chat))
            #print(name,res,chat)
            if res=='positive':
                pos+=1
                opinion[name][0]+=1
            else:
                neg+=1
                opinion[name][1]+=1
        except:
            pass
    print("positive: {} \nNegative: {}".format(pos,neg))
    return opinion, pos, neg

def pie_chart(pos,neg):
    neg = abs(neg) if neg != 0 else 0
    neg=abs(neg)
    labels = ['positive','negative']
    sizes = [pos,neg]
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes ,labels=labels, autopct='%1.1f%%')
    plt.title('Whatsapp Sentiment Analysis')
    return fig1


def bar_plot(opinion):
    names,positive,negative=[],[],[]
    for name in opinion:
        names.append(name)
        positive.append(opinion[name][0])
        negative.append(opinion[name][1])
    ind = np.arange(len(names))
    width=0.3
    max_x=max(max(positive),max(negative))+2

    fig = plt.figure()
    ax = fig.add_subplot()

    yvals = positive
    rects1 = ax.bar(ind, yvals, width, color='g')
    zvals = negative
    rects2 = ax.bar(ind+width, zvals, width, color='r')

    ax.set_xlabel('Names')
    ax.set_ylabel('Sentiment')

    ax.set_xticks(ind+width)
    ax.set_yticks(np.arange(0,max_x,1))
    ax.set_xticklabels( names, rotation=90 )
    ax.legend( (rects1[0], rects2[0]), ('positive', 'negative') )
    ax.set_title('Whatsapp Chat Sentiment Analysis')


    autolabel(rects1, ax)
    autolabel(rects2, ax)

    return fig

def autolabel(rects, ax):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')