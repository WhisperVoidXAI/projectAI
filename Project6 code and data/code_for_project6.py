import nltk
nltk.download('punkt_tab')
import pandas as pd
import string
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from snowballstemmer import stemmer
from wordcloud import WordCloud
import arabic_reshaper
from bidi.algorithm import get_display
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import nltk
nltk.download('punkt')
nltk.download('stopwords') 
ar_stemmer = stemmer("arabic")
def remove_chars(text, del_chars):
    translator = str.maketrans('', '', del_chars)
    return text.translate(translator)

def remove_repeating_char(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)

def cleaningText(text):
    text = re.sub(r'[0-9]+', '', text)
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    text = remove_chars(text, arabic_punctuations + english_punctuations)
    text = remove_repeating_char(text)
    text = text.replace('\n', ' ').strip()
    return text
def tokenizingText(text):
    return word_tokenize(text)
def filteringText(tokens_list):
    stop_words = set(stopwords.words('arabic'))
    return [w for w in tokens_list if w not in stop_words]
def stemmingText(tokens_list):
    return [ar_stemmer.stemWord(word) for word in tokens_list]
def toSentence(words_list):
    return ' '.join(words_list)
tweets = pd.read_csv(r'D:\MyDesktop\ArabicSentimentsAnalysis-HoubAcademy\ArabicSentimentsAnalysis-master\tweets.csv', encoding='utf-8')
positive = pd.read_csv(r'D:\MyDesktop\ArabicSentimentsAnalysis-HoubAcademy\ArabicSentimentsAnalysis-master\positive.csv', encoding='utf-8')
negative = pd.read_csv(r'D:\MyDesktop\ArabicSentimentsAnalysis-HoubAcademy\ArabicSentimentsAnalysis-master\negative.csv', encoding='utf-8')
tweets['tweet_clean'] = tweets['tweet'].apply(cleaningText)
tweets['tweet_preprocessed'] = tweets['tweet_clean'].apply(tokenizingText).apply(filteringText).apply(stemmingText)
tweets.drop_duplicates(subset='tweet_clean', inplace=True)
def preprocess_dictionary(df):
    df['word_clean'] = df['word'].apply(cleaningText)
    df.drop(['word'], axis=1, inplace=True)
    df['word_preprocessed'] = df['word_clean'].apply(tokenizingText).apply(filteringText).apply(stemmingText)
    df.drop_duplicates(subset='word_clean', inplace=True)
    df.dropna(subset=['word_clean'], inplace=True)
    return df
positive = preprocess_dictionary(positive)
negative = preprocess_dictionary(negative)
dict_positive = {row['word_clean'].strip(): int(row['polarity']) for idx, row in positive.iterrows()}
dict_negative = {row['word_clean'].strip(): int(row['polarity']) for idx, row in negative.iterrows()}
def sentiment_analysis_dict_arabic(words_list):
    score = sum(dict_positive.get(word, 0) for word in words_list) + sum(dict_negative.get(word, 0) for word in words_list)
    if score > 0:
        polarity = 'positive'
    elif score < 0:
        polarity = 'negative'
    else:
        polarity = 'neutral'
    return score, polarity
results = tweets['tweet_preprocessed'].apply(sentiment_analysis_dict_arabic)
tweets['polarity_score'], tweets['polarity'] = zip(*results)
fig, ax = plt.subplots(figsize=(6, 6))
x = [count for count in tweets['polarity'].value_counts()]
labels = list(tweets['polarity'].value_counts().index)
explode = [0.1] + [0]*(len(labels)-1)  # يجعل أول شريحة بارزة قليلاً
ax.pie(x=x, labels=labels, autopct='%1.1f%%', explode=explode, textprops={'fontsize': 14})
ax.set_title('Tweets Polarities', fontsize=16, pad=20)
plt.show()
tweets.to_csv(r'D:\MyDesktop\ArabicSentimentsAnalysis-HoubAcademy\ArabicSentimentsAnalysis-master\tweets_clean_polarity.csv', encoding='utf-8', index=False)
def remove_control_characters(text):
    # Unicode: LRI=U+2066, RLI=U+2067, FSI=U+2068, PDI=U+2069
    return re.sub(r'[\u2066\u2067\u2068\u2069]', '', text)
list_words = ' '.join([word for tweet in tweets['tweet_preprocessed'] for word in tweet[:100]])
reshaped_text = arabic_reshaper.reshape(list_words)
reshaped_text = remove_control_characters(reshaped_text)
print("النص بعد إزالة الرموز:", reshaped_text)
try:
    artext = get_display(reshaped_text)
    wordcloud = WordCloud(font_path=r'D:\MyDesktop\ArabicSentimentsAnalysis-HoubAcademy\ArabicSentimentsAnalysis-master\DroidSansMono.ttf', width=600, height=400, background_color='black').generate(artext)
    plt.figure(figsize=(8,6))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
except AssertionError as e:
    print("خطأ في get_display:", e)
    print("تأكد من أن النص لا يحتوي على رموز تحكم أو أحرف غير عربية.")
sentences = tweets['tweet_preprocessed'].apply(toSentence)
max_words = 5000
max_len = 50
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(sentences.values)
X = pad_sequences(tokenizer.texts_to_sequences(sentences.values), maxlen=max_len)
polarity_encode = {'negative':0, 'neutral':1, 'positive':2}
y = tweets['polarity'].map(polarity_encode).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
def create_model(embed_dim=32, hidden_unit=16, dropout_rate=0.2, optimizer=RMSprop, learning_rate=0.001):
    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embed_dim, input_length=max_len))
    model.add(LSTM(units=hidden_unit, dropout=dropout_rate))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer(learning_rate=learning_rate), metrics=['accuracy'])
    print(model.summary())
    return model
model = create_model(embed_dim=32, hidden_unit=64, dropout_rate=0.2, optimizer=Adam, learning_rate=0.001)
history = model.fit(X_train, y_train, validation_split=0.1, epochs=10, batch_size=256, verbose=1)
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall: {recall*100:.2f}%')
print(f'F1 Score: {f1*100:.2f}%')
sns.heatmap(confusion_matrix(y_test, y_pred_classes), annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.show()