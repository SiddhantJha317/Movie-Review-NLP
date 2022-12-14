
# Deep Learning Sentiment Analysis WebApp

This Program attempts to construct a model using tensorflow and keras to analyse the sentiments of movie review using the IMDB Dataset, then uses the same model to construct a prediction based webapp (GUI) employing streamlit in the process.

## Acknowledgements

 - [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
 - [Tensorflow Docs](https://www.tensorflow.org/resources/learn-ml?gclid=CjwKCAiAheacBhB8EiwAItVO2znpaGWFgmraSOgIQCzJBMlDCvE9F-9BtCmFAOyBMUpej1XCbmMevRoCdtUQAvD_BwE)
 - [Streamlit Docs](https://docs.streamlit.io/)

 ## Logic of the Project
 Step 1: Import CSV dataset through pandas.
 ```
 	review	sentiment
0	One of the other reviewers has mentioned that ...	positive
1	A wonderful little production.

The...	positive
2	I thought this was a wonderful way to spend ti...	positive
3	Basically there's a family where a little boy ...	negative
4	Petter Mattei's "Love in the Time of Money" is...	positive
 ```

 Step 2: Clean up the Dataset of all extrenous symbols, extrenous spaces etc.
```
Some sentences have extra spaces , and symbols they must be removed using regex
```

 Step 3: Break up the Dataset into X and Y.
```
Break the csv file into two dataframes of reviews and sentiment.
``` 
 Step 4: Remove all stopwords which don't add any additional context to the review.
```
create a english stopwords variable, When working with text mining applications, 
we often hear of the term “stop words” or “stop word list” or even “stop list”.
Stop words are basically a set of commonly used words in any language, 
not just English.
``` 
 Step 5: Also encode the Y column from 'positive' and 'negative' to 1 and 0.

 Step 6: And finally lower all characters in X to avoid inflections when training the model as capital letters may be (wrongly) recognized as additional context.
```
Lowering is important because it creates inflections in speech which may misinterpreted
by the model as change or shift in context of the sentence.
Through Zip's Law.
```
 Step 7: Use split class to break X and Y into train and test sets.

 Step 8: Now we must tokenize the X_train and X_test array.
```
tokenization is a process where words inside passages and sentence are converted into a 
relational set of numbers , which are ontologically self referential and hence can be used
to normalize and hence can be pushed through the LSTM model.

Tokenization is a way of separating a piece of text into smaller units called tokens. 
Here, tokens can be either words, characters, or subwords. Hence, 
tokenization can be broadly classified into 3 types – word, character, 
and subword (n-gram characters) tokenization.

For example, consider the sentence: “Never give up”.

The most common way of forming tokens is based on space. Assuming space as a delimiter,
the tokenization of the sentence results in 3 tokens – Never-give-up. As each token is a word,
it becomes an example of Word tokenization.
```
 Step 9: Initialize the tokenizer , both truncate and padding.

 Step 10: Construct the LSTM model using keras. Sequential ofcourse.
```
LSTM stands for long short-term memory networks, used in the field of Deep Learning.
It is a variety of recurrent neural networks (RNNs) that are capable of learning long-term dependencies, 
especially in sequence prediction problems. LSTM has feedback connections, i.e.,
it is capable of processing the entire sequence of data, apart from single data points such as images.
This finds application in speech recognition, machine translation, etc.
LSTM is a special kind of RNN, which shows outstanding performance on a large variety of problems.
```
 Step 11: fit the model and save it as 'h5' file.
```
Save the file to re access it when constructing the WebApp , tensorflow encodes the model in h5 format.
```



## File System(Orginal)
```
NLP_Sentiment
|
|-----IMDB Dataset.csv
|-----model.ipynb
|-----app.py
|-----requirements.txt
|-----model.h5
```

## Code
The Code is divided into a pure python 3.10 file '.py' which is named 'app.py' and is used to deploy the streamlit webapp, the 'model.ipynb' is jupyter kernel which is used to both preprocess and train the model.

### model.ipynb
Importing all the necessary libraries
```
import pandas as pd    
import numpy as np     
import nltk
from nltk.corpus import stopwords   
from sklearn.model_selection import train_test_split      
from tensorflow import keras 
from keras.preprocessing.text import Tokenizer  
from keras.utils import pad_sequences   
from keras.models import Sequential     
from keras.layers import Embedding, LSTM, Dense 
import re
```
Read in the file as a csv 

```
data = pd.read_csv("IMDB Dataset.csv")
data.head(5)
```
Set up english stopwords variable
```
nltk.download('stopwords')
english_stops = set(stopwords.words('english'))
```
Now Clean and break data into x_data and y_data.
```
# Read in the csv file through pandas
df = pd.read_csv("IMDB Dataset.csv")
x_data = df['review']
y_data = df['sentiment']

# Clean data through regex , via removing all symbols
x_data = x_data.replace({'<.*?': ''}, regex=True)
# Replace extraneous spaces with words next to them
x_data = x_data.replace({'[^A-Za-z]': ' '}, regex = True)
#  Now remove all stopwords eg: a , are , the      
x_data = x_data.apply(lambda review: [w for w in review.split() if w not in english_stops])  # remove stop words
# lower all the words
x_data = x_data.apply(lambda review: [w.lower() for w in review]) 
# encode and replace positive and negative with 1 and 0.
y_data = y_data.replace('positive', 1)

y_data = y_data.replace('negative', 0)

print('Reviews')
print(x_data, '\n')
print('Sentiment')
print(y_data)
-------------------------------------------------------------------------------------------
Reviews
0        [one, reviewers, mentioned, watching, oz, epis...
1        [a, wonderful, little, production, br, br, the...
2        [i, thought, wonderful, way, spend, time, hot,...
3        [basically, family, little, boy, jake, thinks,...
4        [petter, mattei, love, time, money, visually, ...
                               ...                        
49995    [i, thought, movie, right, good, job, it, crea...
49996    [bad, plot, bad, dialogue, bad, acting, idioti...
49997    [i, catholic, taught, parochial, elementary, s...
49998    [i, going, disagree, previous, comment, side, ...
49999    [no, one, expects, star, trek, movies, high, a...
Name: review, Length: 50000, dtype: object 

Sentiment
0        1
1        1
2        1
3        0
4        1
        ..
49995    1
49996    0
49997    0
49998    0
49999    0
Name: sentiment, Length: 50000, dtype: int64
```
Split into training and test sets
```
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2)
```
Get the max length avg to restrict the control preprocess
```
def get_max_length():
    review_length = []
    for review in x_train:
        review_length.append(len(review))

    return int(np.ceil(np.mean(review_length)))
a = get_max_length()
print(a)
----------------------------------------------
134
```
Set up the tokenization system
```
token = Tokenizer(lower=False)    # no need lower, because already lowered the data in load_data()
token.fit_on_texts(x_train)
x_train = token.texts_to_sequences(x_train)
x_test = token.texts_to_sequences(x_test)

max_length = get_max_length()

x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')

total_words = len(token.word_index) + 1   # add 1 because of 0 padding
---------------------------------------------------------------------
Encoded X Train
 [[ 309 2250   22 ...    0    0    0]
 [   2  658 1844 ...    0    0    0]
 [   2  889    2 ...    0    0    0]
 ...
 [ 205   24 4206 ...    0    0    0]
 [   2  102   29 ...    7   85  781]
 [   9  228    5 ...    0    0    0]] 

Encoded X Test
 [[  40 1899  300 ...    0    0    0]
 [ 623  337 1635 ...    0    0    0]
 [3225  124    3 ...    0    0    0]
 ...
 [  34    4   15 ...    0    0    0]
 [  79  167    7 ...    0    0    0]
 [   3  611   14 ...    0    0    0]] 
```
Build the LSTM deep learning model with keras
```
EMBED_DIM = 32
LSTM_OUT = 64

model = Sequential()
model.add(Embedding(total_words, EMBED_DIM, input_length = max_length))
model.add(LSTM(LSTM_OUT))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

print(model.summary())
----------------------------------------------------------------------
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 134, 32)           2934752   
                                                                 
 lstm (LSTM)                 (None, 64)                24832     
                                                                 
 dense (Dense)               (None, 1)                 65        
                                                                 
=================================================================
Total params: 2,959,649
Trainable params: 2,959,649
Non-trainable params: 0
_________________________________________________________________
None
```
Fit the model
```
model.fit(x_train, y_train, batch_size = 128, epochs = 5)
----------------------------------------------------------------------
Epoch 1/5
313/313 [==============================] - 42s 126ms/step - loss: 0.5187 - accuracy: 0.7018
Epoch 2/5
313/313 [==============================] - 39s 126ms/step - loss: 0.2337 - accuracy: 0.9146
Epoch 3/5
313/313 [==============================] - 40s 128ms/step - loss: 0.1391 - accuracy: 0.9567
Epoch 4/5
313/313 [==============================] - 41s 132ms/step - loss: 0.0898 - accuracy: 0.9747
Epoch 5/5
313/313 [==============================] - 41s 131ms/step - loss: 0.0637 - accuracy: 0.9829
```
Save the Model
```
model.save('model.h5')
```
### App.py
Importing libraries into the file
```
import re
import keras
import nltk
from nltk.corpus import stopwords   
nltk.download('stopwords')
from keras.preprocessing.text import Tokenizer  
from keras.utils import pad_sequences   
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split    
```
Reruning the cleaning process of X and  Y in order to fit the new tokenizer(necessary)
```
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
```
Now through streamlit design the web app set up the heading and subheading
```
st.title("Movie Review Sentiment Analysis")
st.write("This ML model will guess if a given review is positive or negative using LSTM "
         "This model was trained using Tensorflow and on the Long Short Term Memory RNN")
```
Run the streamlit prediction script and set up up output variables and dependencies
```
if text != '':    # check for non empty text box
   if st.button('Analyse'): # check if the button is clicked
    loaded = keras.models.load_model('D:\\NLP_Sentiment\\model.h5') # load the model
    regex = re.compile(r'[^a-zA-Z\s]')

    words = text.split(' ')
    newtext = [w for w in words if w not in english_stops]
    newtext = ' '.join(newtext)
    newtext = [newtext.lower()]

    tokenize_words = token.texts_to_sequences(newtext)
    tokenize_words = pad_sequences(tokenize_words, maxlen=134, padding='post', truncating='post')
    answer = loaded.predict(tokenize_words)
    if answer > [[0.6]]: # if the probability is greater than 0.6 then the review is positive
        st.write(f'positive with probability {answer[0]}')
    else:
        st.write(f'negative with probability {answer[0]}')
```


## Output
Positive Example:

![image1](https://user-images.githubusercontent.com/111745916/207584818-223c9bff-0396-4ba7-86cf-05dccd08ee06.png)


Negative Example:

![image2](https://user-images.githubusercontent.com/111745916/207584867-b7ec05f0-a357-43e4-841b-b700030e3272.png)


Video Example:




## Requirements and dependencies
```
requirements.txt
|
|-------keras<=2.11.0
|-------nltk<=3.7
|-------pandas<=1.4.4
|-------scikit_learn<=1.2.0
|-------streamlit<=1.12.2
|-------tensorflow<=2.11.0
|-------tensorflow_intel<=2.11.0
```
## Testing


```
clone the repository to your local machine
```

```
unzip to new folder
```
```
install python >= 3.9
```
```
install dependencies via requirements.txt
```
Change the file paths in app.py to your correspoding data files
```
df = pd.read_csv("D:\\NLP_Sentiment\\IMDB Dataset.csv")
```
Now either run model.ipynb and it will create a model.h5 file in your local folder to 
be used in the webapp.
```
run /yourpath/model.ipynb
```
you may also run the above in google colab if you don't have jupyter locally.

Finally run this command
```
streamlit run /yourpath/app.py
```
This will direct you to your browser if it doesn't copy local https address displayed by them into your local browser
https connection may not be required.

### Thanks for Reading!!!
