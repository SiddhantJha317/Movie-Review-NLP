import re
import keras
from nltk.corpus import stopwords   
nltk.download('stopwords')
from keras.preprocessing.text import Tokenizer  
from keras.utils import pad_sequences   
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split      

english_stops = set(stopwords.words('english'))

df = pd.read_csv("D:\\NLP_Sentiment\\IMDB Dataset.csv")
x_data = df['review']
y_data = df['sentiment']

x_data = x_data.replace({'<.*?': ''}, regex=True)
x_data = x_data.replace({'[^A-Za-z]': ' '}, regex = True)     
x_data = x_data.apply(lambda review: [w for w in review.split() if w not in english_stops])  # remove stop words
x_data = x_data.apply(lambda review: [w.lower() for w in review]) 
y_data = y_data.replace('positive', 1)

y_data = y_data.replace('negative', 0)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2)
token = Tokenizer(lower=False)
token.fit_on_texts(x_train)








st.title("Movie Review Sentiment Analysis")
st.write("This ML model will guess if a given review is positive or negative using LSTM "
         "This model was trained using Tensorflow and on the Long Short Term Memory RNN")


text = st.text_input("Enter your review")

if text != '':
   if st.button('Analyse'): 
    loaded = keras.models.load_model('D:\\NLP_Sentiment\\model.h5')
    regex = re.compile(r'[^a-zA-Z\s]')

    words = text.split(' ')
    newtext = [w for w in words if w not in english_stops]
    newtext = ' '.join(newtext)
    newtext = [newtext.lower()]

    tokenize_words = token.texts_to_sequences(newtext)
    tokenize_words = pad_sequences(tokenize_words, maxlen=134, padding='post', truncating='post')
    answer = loaded.predict(tokenize_words)
    if answer > [[0.6]]:
        st.write(f'positive with probability {answer[0]}')
    else:
        st.write(f'negative with probability {answer[0]}')
