

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')

df

df.info()

df.isnull().sum()

df.dropna(inplace=True)

df.describe()

# df['Sentiment'] = df['Ratings'].apply(lambda x: 1 if x > 3 else 0)
!pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

def label_sentiment(review_text):
    # Analyze sentiment using VADER
    sentiment_score = analyzer.polarity_scores(review_text)

    # Determine sentiment label based on compound score
    sentiment_label = 1 if sentiment_score['compound'] >= 0 else 0

    return sentiment_label

# Apply label_sentiment function to the 'Review text' column
df['Sentiment'] = df['Review text'].apply(label_sentiment)

df

"""**Distribution Analysis**"""

# Analyze distribution of ratings
plt.hist(df['Ratings'], bins=5, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Analyze distribution of ratings over time
df['Month'] = pd.to_datetime(df['Month'])  # Convert 'Month' column to datetime format
df['YearMonth'] = df['Month'].dt.to_period('M')  # Extract Year-Month
ratings_over_time = df.groupby('YearMonth')['Ratings'].mean()

plt.figure(figsize=(10, 6))
ratings_over_time.plot(marker='o', color='green', linestyle='-', markersize=8)
plt.title('Average Ratings Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Average Rating')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

"""**Identify Input and Output**"""

X = df['Review text']
y = df['Sentiment']

"""**Split the Data into Train and Test**"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

"""**Data Preprocessing on train data (X_train)**"""

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def display_wordcloud(data):
    wc = WordCloud(background_color='black',
               width=1600,
               height=800).generate(' '.join(data))
    plt.figure(1,figsize=(30,20))
    plt.imshow(wc)
    plt.axis('off')
    plt.show()

display_wordcloud(X_train[y_train==1])

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')

nltk.download('punkt')

nltk.download('wordnet')

def clean(doc): # doc is a string of text

    doc = re.sub(r'[^\w\s]', '', doc)

    # Remove punctuation and numbers.
    doc = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])

    # Converting to lower case
    doc = doc.lower()

    # Tokenization
    tokens = nltk.word_tokenize(doc)

    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Stop word removal
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in lemmatized_tokens if word.lower() not in stop_words]

    # Join and return
    return " ".join(filtered_tokens)

# Commented out IPython magic to ensure Python compatibility.
# import feature extraction methods from sklearn
from sklearn.feature_extraction.text import CountVectorizer

# instantiate a vectorizer
vect = CountVectorizer(preprocessor=clean)

# use it to extract features from training data
# %time X_train_dtm = vect.fit_transform(X_train)

print(X_train_dtm.shape)

"""**Data Preprocessing on test data (X_test)**"""

# transform testing data (using training data's features)
X_test_dtm = vect.transform(X_test)

print(X_test_dtm.shape)

""" **Building a Model (i.e. Train the classifier)**"""

# Commented out IPython magic to ensure Python compatibility.
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB() # instantiate a Multinomial Naive Bayes model
# %time nb.fit(X_train_dtm, y_train) # train the model(timing it with an IPython "magic command")

"""**Evaluating on Train Data**"""

from sklearn import metrics
# make class predictions for X_train_dtm
y_train_pred = nb.predict(X_train_dtm)
metrics.accuracy_score(y_train, y_train_pred)

metrics.confusion_matrix(y_train, y_train_pred)

cm = metrics.confusion_matrix(y_train, y_train_pred)
actual = np.sum(cm, axis=1).reshape(-1, 1)
cmn = np.round(cm/actual, 2)

sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=nb.classes_, yticklabels=nb.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')

"""**Evaluate on Test Data**"""

# make class predictions for X_test_dtm
y_test_pred = nb.predict(X_test_dtm)

metrics.accuracy_score(y_test, y_test_pred)

cm = metrics.confusion_matrix(y_test, y_test_pred)
actual = np.sum(cm, axis=1).reshape(-1, 1)
cmn = np.round(cm/actual, 2)

sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=nb.classes_, yticklabels=nb.classes_)
plt.ylabel('Actual')
plt.xlabel('Predicted')

print("Train Score (F1 Score):", metrics.f1_score(y_train, y_train_pred))

print("Test Score (F1 Score):", metrics.f1_score(y_test, y_test_pred))

"""**Creating an Optimal Workflow**"""

# Commented out IPython magic to ensure Python compatibility.
# %time X_train_clean = X_train.apply(lambda doc: clean(doc))

# Commented out IPython magic to ensure Python compatibility.
# %time X_test_clean = X_test.apply(lambda doc: clean(doc))

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics


from sklearn.pipeline import Pipeline
import warnings
import joblib
from joblib import Memory
import os
warnings.filterwarnings('ignore')

# Commented out IPython magic to ensure Python compatibility.
# Define a memory object to cache intermediate results
cachedir = '.cache'
memory = Memory(location=cachedir, verbose=0)

pipelines = {
    'naive_bayes': Pipeline([
        ('vectorization', CountVectorizer()),
        ('classifier', MultinomialNB())
    ], memory=memory),
    'decision_tree': Pipeline([
        ('vectorization', CountVectorizer()),
        ('classifier', DecisionTreeClassifier())
    ], memory=memory),
    'logistic_regression': Pipeline([
        ('vectorization', CountVectorizer()),
        ('classifier', LogisticRegression())
    ], memory=memory)
}

# Define parameter grid for each algorithm
param_grids = {
    'naive_bayes': [
        {
            'vectorization': [CountVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000],
            'classifier__alpha' : [1, 10]
        }
    ],
    'decision_tree': [
        {
            'vectorization': [CountVectorizer(), TfidfVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000],
            'classifier__max_depth': [None, 5, 10]
        }
    ],
    'logistic_regression': [
        {
            'vectorization': [CountVectorizer(), TfidfVectorizer()],
            'vectorization__max_features' : [1000, 1500, 2000, 5000],
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['elasticnet'],
            'classifier__l1_ratio': [0.4, 0.5, 0.6],
            'classifier__solver': ['saga'],
            'classifier__class_weight': ['balanced']
        }
    ]
}

# Perform GridSearchCV for each algorithm
best_models = {}

for algo in pipelines.keys():
    print("*"*10, algo, "*"*10)
    grid_search = GridSearchCV(estimator=pipelines[algo],
                               param_grid=param_grids[algo],
                               cv=5,
                               scoring='f1',
                               return_train_score=True,
                               verbose=1
                              )

#     %time grid_search.fit(X_train_clean, y_train)

    best_models[algo] = grid_search.best_estimator_

    print('Score on Test Data: ', grid_search.score(X_test_clean, y_test))

for name, model in best_models.items():
    print(f"{name}")
    print(f"{model}")
    print()

from google.colab import drive
drive.mount('/content/drive')
save_path = '/content/drive/My Drive/best_models_2/'

# Commented out IPython magic to ensure Python compatibility.
for name, model in best_models.items():
    print("*"*10, name, "*"*10)

    joblib.dump(model, save_path + f'{name}.pkl')
    model = joblib.load(save_path + f'{name}.pkl')

#     %time y_test_pred = model.predict(X_test_clean)
    print("Test Score (F1)", metrics.f1_score(y_test, y_test_pred))

    print("Model Size:", os.path.getsize(save_path + f'{name}.pkl'), "Bytes")

