import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras._tf_keras.keras.layers import Dense
from keras._tf_keras.keras.models  import Sequential
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import pickle

data = pd.read_csv(r'C:\Users\salwa\OneDrive\Desktop\machine learning projects\Emotion Detection with NN\Data\text.csv')
print(data.head())
data = data.iloc[:, 1:]
data.to_csv('new_text.csv',index=False)

label_mapping = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
data2 = pd.read_csv(r'C:\Users\salwa\OneDrive\Desktop\machine learning projects\Emotion Detection with NN\new_text.csv')
data2['label'] = data['label'].replace(label_mapping)
data2.to_csv('new_text.csv',index=False)

df = pd.read_csv(r'C:\Users\salwa\OneDrive\Desktop\machine learning projects\Emotion Detection with NN\new_text.csv')
print(df['label'].value_counts())
plt.hist(x='label', bins=12, data=df, color='indigo' )
plt.title('Data Distribution of different Emotions')
plt.xlabel('Emotions')
plt.ylabel('Frequency')
plt.savefig('emotion_distribution.png')
plt.show()

# # PreProcessing Data 
def preprocess(text):

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    stopwords_set = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word.lower() not in stopwords_set])
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text

df['text'] = df['text'].apply(preprocess)
encoder = preprocessing.LabelEncoder()

df['label'] = encoder.fit_transform(df['label']) 
df['text'].fillna('', inplace=True)
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
cv.fit(df['text'])
X = cv.fit_transform(df['text']).toarray()
#Splitting the data
y = df['label']
X_train, X_test ,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from keras._tf_keras.keras.layers import Dense, Dropout, BatchNormalization

# Improved model architecture
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    BatchNormalization(),  
    Dropout(0.3),          

    Dense(48, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(24, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(12, activation='relu'),
    BatchNormalization(),

    Dense(6, activation='softmax')  
])

# Compile the model with a better optimizer and evaluation metric
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, y_train, epochs=10 , batch_size=10, validation_data=(X_test, y_test))
# Plotting accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("model accuracy.png")
plt.show()

# Plotting loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('Model Loss.png')
plt.show()

plt.show()

tf.keras.models.save_model(model, 'my_model.h5')
with open("count_vectorizer.pkl", "wb") as f:
    pickle.dump(cv, f)
pickle.dump(encoder, open('encoder.pkl', 'wb'))
loss , accuracy = model.evaluate(X_train, y_train)
print("the accuracy and loss in testing data is ", accuracy, loss)
