import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Text processing
import wikipedia
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from langdetect import detect

# Clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Functions
from algorithms import *
from config import PARAMS


# = = = = = = = = = = = = = = = = = = = = = = = = = = 


# list of the languages of the randomly generated articles
languages = PARAMS.LANGUAGES

# dictionnary of the language codes and the full name of the languages
lang_dict = PARAMS.LANG_DICT

# number of documents to generate per language
articles_per_lang = PARAMS.ARTICLES_PER_LANG

# number of clusters to generate for each language group
nclusters = PARAMS.NCLUSTERS

# number of keywords per cluster
n_terms = PARAMS.NTERMS

# plot 2D graph of the clusters per language
plot2d = PARAMS.PLOT2D


# = = = = = = = = = = = = = = = = = = = = = = = = = = 

def main():
    
    ### Document Extraction
    print("\n","*"*20,"Document Extraction","*"*20,'\n')
    print("Generating Random Wikipedia articles...")  
    articles = random_articles(languages, articles_per_lang)
    print(len(articles), "articles generated!")
    
    
    ### Language Detection
    print("\n","*"*20,"Language Detection","*"*20,'\n')
    print("Detecting languages...") 
    articles_dict = detect_language(articles, len(languages))

    for lang in articles_dict.keys():
        print("language: ",lang_dict[lang],", N° documents:", len(articles_dict[lang]),", Avg document length",
              int(np.mean([len(article) for article in articles_dict[lang]])))

    
    ### Text Processing
    print("\n","*"*20,"Text Processing","*"*20,'\n')
    print("Generating tokens...")
    tokens_dict = {}
    for lang in articles_dict.keys():
        articles = []
        for article in articles_dict[lang]:
            articles.append(process_article(article, lang))
        tokens_dict[lang] = articles
    print("Tokens generated!")
    
    
    ### Document Clustering
    print("\n","*"*20,"Document Clustering","*"*20,'\n')
    print("Generating TF-IDF matrix...\n")
    tfidf_articles = {}
    tfidf_labels = {}
    
    for lang in articles_dict.keys():
        tfidf_articles[lang], tfidf_labels[lang] = tfidf_features(tokens_dict[lang])
        print(lang_dict[lang],' articles features: ', tfidf_articles[lang].shape)
        
    print("\nGenerating clusters...\n")
    labels = {}
    for lang in articles_dict.keys():
        labels[lang] = cluster_documents(tfidf_articles[lang], nclusters[lang])
    
    for lang in labels.keys():
        print('\n',lang_dict[lang], 'articles group:')
        for cluster in Counter(labels[lang]).keys():
            print('cluster',cluster,': ',Counter(labels[lang])[cluster],'articles')
    
    top_keywords = {}
    
    for lang in articles_dict.keys():
        top_keywords[lang] = top_keywords_per_cluster(tfidf_articles[lang].toarray(), 
                                                      labels[lang], tfidf_labels[lang], n_terms)
    print("\n=====Top keywords per cluster=====\n")
    for lang in top_keywords.keys():
        print('\n',lang_dict[lang],'language')
        for i in range(len(top_keywords[lang])):
            keywords = ', '.join(top_keywords[lang][i])
            print('Cluster',i,':',keywords)
    
    print("\n=====Silhouette score=====\n")
    for lang in tfidf_articles.keys():
        print(lang_dict[lang],'articles clustering score', 
              round(silhouette_score(tfidf_articles[lang], labels[lang]),2))
    
    # export reults to a csv file
    results = []
    for lang in articles_dict.keys():
        for i in range(len(articles_dict[lang])):
            temp = []
            temp.append(articles_dict[lang][i])
            temp.append(lang_dict[lang])
            temp.append(labels[lang][i])
            results.append(temp)

    pd.DataFrame(results, columns = ['article', 'language', 'cluster']).to_csv('results.csv')

    if plot2d:
        print("\n=====2D plot of clusters=====\n")
        for lang in articles_dict.keys():
            plot_clusters(tfidf_articles[lang].toarray(), labels[lang], lang)    
        
        
if __name__== "__main__":
  main()


