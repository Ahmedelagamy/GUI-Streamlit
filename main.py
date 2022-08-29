# -*- coding: utf-8 -*-
"""

@author: Ahmed El Agamy
"""
import pandas as pd
# Imports
import streamlit as st
from bertopic import BERTopic
from textblob import TextBlob
from umap import UMAP


# Functions for interpreting TextBlob analysis
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# Create a function to get the polarity
def get_polarity(text):
    return TextBlob(text).sentiment.polarity


# Applying Analysis Function
def get_analysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'


# Code for project structure
st.image('piovos_logo.png')
st.title("Piovis Automate")
st.sidebar.title('Review analyzer GUI')
st.markdown("This application is a streamlit deployment to automate analysis")

file = st.file_uploader('upload the dataset you want to analyze')

# Data preprocessing
# noinspection PyArgumentList
df = pd.read_excel(file)
df.dropna(subset=['review-text'], inplace=True)
df.drop(index=1637, axis=0, inplace=True)

# Displaying DataFrame
st.dataframe(df)

# Splitting data
bad_reviews = df[df['TextBlob_Analysis'] == 'Negative']
good_reviews = df[df['TextBlob_Analysis'] == 'Positive']

# Minor mod
st.header('Select Stop Words')

custom_stopwords = st.text_input('enter stop word')


def clean_text(dataframe, col_name):
    import re
    import nltk
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    nltk.download('wordnet')
    from nltk.stem.wordnet import WordNetLemmatizer
    lem = WordNetLemmatizer()
    stem = PorterStemmer()
    word = "inversely"
    print("stemming:", stem.stem(word))
    print("lemmatization:", lem.lemmatize(word, "v"))

    # Creating a list of stop words and adding custom stopwords
    stop_words = set(stopwords.words("english"))

    # Creating a list of custom stopwords
    new_words = custom_stopwords.split(',')
    stop_words = stop_words.union(new_words)

    docs = []
    for i in dataframe[col_name]:
        # Remove punctuations
        text = re.sub('[^a-zA-Z]', ' ', i)

        # Convert to lowercase
        text = text.lower()

        # remove tags
        text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

        # remove special characters and digits
        text = re.sub("(\\d|\\W)+", " ", text)

        # Convert to list from string
        text = text.split()

        # Stemming
        PorterStemmer()
        # Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text if word not in stop_words]

        text = " ".join(text)
        # print(text)
        docs.append(text)
    # print(docs)
    return docs


# Applying function
good_reviews = clean_text(good_reviews, 'review-text')
bad_reviews = clean_text(bad_reviews, 'review-text')
final_df= df.groupby(['asin', 'product-name', 'rating-count', 'rating-avg', 'TextBlob_Analysis','detect']).count()
# Tab Structure
tab = st.sidebar.radio('Select one:', ['Positive Review', 'Negative Review'])

# Models
if tab == 'Positive Review':
    st.subheader('Positive Reviews')
    st.dataframe(df[df['TextBlob_Analysis'] == 'Positive']['review-text'])
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
    topic_model_1 = BERTopic(diversity=.9, embedding_model='paraphrase-MiniLM-L3-v2', verbose=True,
                             calculate_probabilities=True, nr_topics=15, umap_model=umap_model)

    good_model = topic_model_1.fit(good_reviews)

    good_model.generate_topic_labels(nr_words=4)

    """# Good Reviews model insight"""

    doc_num = int(st.number_input('enter the number of topic to explore'))

    st.write(good_model.get_topic_info())

    st.write(good_model.get_representative_docs(doc_num))

    st.write(good_model.visualize_topics())

    st.write(good_model.visualize_barchart())

    st.write(good_model.visualize_heatmap())
    # pros
    good_topic_info = good_model.get_topic_info()
    good_topic_info['percentage'] = good_topic_info['Count'].apply(lambda x: (x / good_topic_info['Count'].sum()) * 100)
    st.write(good_topic_info)

else:

    """# Bad reviews model insight"""
    # Feature Engineering
    st.subheader('Negative Reviews')
    st.dataframe(df[df['TextBlob_Analysis'] == 'Negative']['review-text'])
    topic_model_2 = BERTopic(embedding_model='paraphrase-MiniLM-L3-v2', verbose=True, nr_topics='auto',
                             calculate_probabilities=True)

    bad_model = topic_model_2.fit(bad_reviews)
    # Topics
    st.write(bad_model.get_topic_info())
    doc_num = int(st.number_input('enter the number of topic to explore'))
    # Labels
    st.write(bad_model.generate_topic_labels(nr_words=6, separator=", "))
    # Representative docs
    st.write(bad_model.get_representative_docs(doc_num))
    # Topic visualization
    # Bar chart
    st.write(bad_model.visualize_barchart())
    # Term Rank
    st.write(bad_model.visualize_term_rank())
    # Heatmap
    bad_model.visualize_heatmap()
    # cons
    bad_model.generate_topic_labels(nr_words=6, separator=',')
    bad_model.get_representative_docs(doc_num)

    bad_topic_info = bad_model.get_topic_info()
    bad_topic_info['percentage'] = bad_topic_info['Count'].apply(lambda x: (x / bad_topic_info['Count'].sum()) * 100)
    st.write(bad_topic_info)


final_df.drop(['TextBlob_Subjectivity','TextBlob_Polarity', 'Unnamed: 0','review-text'], axis= 1, inplace = True)

st.write(final_df)
