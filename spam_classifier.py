import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from sklearn.metrics import confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt
import seaborn as sns

veri = pd.read_csv('spam.csv')

X = veri["v2"].values
y = veri["v1"]
y = y.replace("ham",0.).replace("spam",1.)

_stopwords = list(stopwords.words('English'))
def temizle(X):
    texts_cleaned = []
    for text in X:
        i = str(text)
        i = i.lower()
        i = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",i).split())
        i = re.sub(r'\W', ' ', str(i))
        i = re.sub(r'\<a href', ' ', i)
        i = re.sub(r'&amp;', '', i)
        i = re.sub(r'<br />', ' ', i)
        i = re.sub(r"^\s+|\s+$", "", i)
        i = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', i)
        i = re.sub(r'\'', ' ', i)
        i = i.split()
        i = [word for word in i if word not in _stopwords]
        i = ' '.join(i)
        texts_cleaned.append(i)
    return texts_cleaned

X = temizle(X)

num_words = 8000
tok = Tokenizer(num_words=num_words)
tok.fit_on_texts(X)
sequences = tok.texts_to_sequences(X)
X_new = sequence.pad_sequences(sequences, padding = "post", truncating ="post")

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.15, random_state = 42)

ysa = Sequential()
ysa.add(Embedding(num_words, 56, input_length=X_new.shape[1]))
ysa.add(Flatten())
ysa.add(Dense(72))
ysa.add(Activation("tanh"))
ysa.add(Dropout(0.2))
ysa.add(Dense(48))
ysa.add(Activation("relu"))
ysa.add(Dense(1))
ysa.add(Activation("sigmoid"))

ysa.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

ysa.summary()

ysa.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))
test_loss, test_acc = ysa.evaluate(X_test, y_test)

ysa_tahmin = ysa.predict(X_test)
ysa_binary_tahmin = [1 if i > 0.5 else 0 for i in ysa_tahmin]

sns.heatmap(confusion_matrix(y_test,ysa_binary_tahmin), annot = True, cmap = "PRGn")
plt.show()
