import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

## DATA PREPARATION
df = pd.read_csv('./Tweets.csv')
df.head()

# Convert text to lowercase
df['text'] = df['text'].str.lower()

df['text'] = df['text'].astype(str)  # Convert 'text' column to string data type

df['tokens'] = df['text'].apply(nltk.word_tokenize)  # Tokenization

# Remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stopwords])

X = df['text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## FEATURE EXTRACTION
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

## BUILD & TRAIN MODEL
model = SVC()
model.fit(X_train_vectors, y_train)

## EVALUATE THE MODEL
y_pred = model.predict(X_test_vectors)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))
