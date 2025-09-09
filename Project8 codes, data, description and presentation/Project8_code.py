# -*- coding: utf-8 -*-
import nltk
nltk.download('stopwords')
import pandas as pd
import string
import re
from nltk.corpus import stopwords
from snowballstemmer import stemmer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.optimizers import Adam, RMSprop
from imblearn.over_sampling import RandomOverSampler

# ============================
# قراءة البيانات
# ============================
tweets = pd.read_csv('tweets.csv', encoding="utf-8")
print('Data size:', tweets.shape)

# ============================
# إعدادات معالجة النصوص
# ============================
ar_stemmer = stemmer("arabic")

def remove_chars(text, del_chars):
    translator = str.maketrans('', '', del_chars)
    return text.translate(translator)

def remove_repeating_char(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)

def clean_tweet(tweet):
    stop_words = stopwords.words('arabic')
    tweet = str(tweet)
    tweet = re.sub(r"RT\s?", "", tweet)
    tweet = re.sub(r"@[^\s]+", "", tweet)
    tweet = re.sub(r"(?:http[s]?://|www\.)\S+", "", tweet)
    tweet = tweet.replace("#", " ").replace("_", " ")
    tweet = re.sub(r'\d+', '', tweet)
    arabic_punctuations = '''`÷× ؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    tweet = remove_chars(tweet, arabic_punctuations + string.punctuation)
    tweet = remove_repeating_char(tweet)
    tweet = tweet.replace('\n', ' ').strip()
    return tweet

def tokenizingText(text):
    return text.split()

def stemmingText(tokens_list):
    return [ar_stemmer.stemWord(word) for word in tokens_list]

def toSentence(words_list):
    return ' '.join(words_list)

def process_tweet(tweet):
    tweet = clean_tweet(tweet)
    tweet = tokenizingText(tweet)
    tweet = stemmingText(tweet)
    return tweet

tweets['tweet'] = tweets['tweet'].apply(process_tweet)

# ============================
# Oversampling للتوازن
# ============================
plt.figure(figsize=(12, 6))
sns.countplot(data=tweets, y='topic')
plt.title('Topics Distribution', fontsize=18)
plt.show()

oversample = RandomOverSampler()
tweets = tweets.sample(frac=1)
tweets, Y = oversample.fit_resample(tweets, tweets.topic)

plt.figure(figsize=(12, 6))
sns.countplot(data=tweets, y='topic')
plt.title('Topics Distribution After OverSampling', fontsize=18)
plt.show()

# ============================
# ترميز الأصناف
# ============================
le_topics = LabelEncoder()
tweets['topic'] = tweets[['topic']].apply(le_topics.fit_transform)
classes = le_topics.classes_
n_classes = len(classes)

# ============================
# إعداد النصوص للتعلم العميق
# ============================
sentences = tweets['tweet'].apply(toSentence)
max_words = 5000
max_len = 50
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(sentences)
S = tokenizer.texts_to_sequences(sentences)
X = pad_sequences(S, maxlen=max_len)
y = tweets['topic']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# ============================
# بناء النموذج
# ============================
def create_model(embed_dim=32, hidden_unit=64, dropout_rate=0.2,
                 optimizers=Adam, learning_rate=0.001):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embed_dim, input_length=max_len))
    model.add(LSTM(units=hidden_unit, dropout=dropout_rate))
    model.add(Dense(units=len(classes), activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers(learning_rate=learning_rate),
                  metrics=['accuracy'])
    print(model.summary())
    return model

model = create_model()

# ============================
# تدريب النموذج
# ============================
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.1
)

# ============================
# تقييم النموذج
# ============================
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

print('Model Accuracy on Test Data:', accuracy * 100)
print('Model Precision on Test Data:', precision * 100)
print('Model Recall on Test Data:', recall * 100)
print('Model F1 on Test Data:', f1 * 100)

# ============================
# تصنيف أي شخص من ملف CSV
# ============================
def classify_tweet(processed_tweet):
    sentence = toSentence(processed_tweet)
    seq = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=max_len)
    return model.predict(padded)[0]

def classify_person(person_name):
    path = person_name + '.csv'
    df = pd.read_csv(path)

    classes_count = {key: 0 for key in classes}
    total = 0

    for _, row in df.iterrows():
        tweet = row['tweet']
        processed_tweet = process_tweet(tweet)

        if len(processed_tweet) > 0:
            try:
                c = classify_tweet(processed_tweet)
                idx_label = c.argmax()
                topic = le_topics.inverse_transform([idx_label])[0]
                classes_count[topic] += 1
                total += 1
            except Exception as e:
                print("Error classifying tweet:", e)
                continue

    sorted_classes = sorted(classes_count, key=classes_count.get, reverse=True)
    sorted_classes_cleaned = {w: classes_count[w] for w in sorted_classes if classes_count[w] > 0}

    n = sum(sorted_classes_cleaned.values()) or 1

    print(person_name, "is classified as :")
    for key, value in sorted_classes_cleaned.items():
        print(key, "(", "{:.2f}".format((value / n) * 100), "%)")

    x = list(sorted_classes_cleaned.keys())
    y = list(sorted_classes_cleaned.values())
    plt.figure(figsize=(9,9))
    plt.title(person_name, fontdict={'fontsize':20})
    plt.pie(y, labels=x, autopct='%1.1f%%')
    plt.show()
classify_person(r"D:\MyDesktop\project7\salem2")
