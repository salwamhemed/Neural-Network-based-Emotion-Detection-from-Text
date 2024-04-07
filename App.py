import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf 
from preprocess import *


encoder = pickle.load(open('encoder.pkl', 'rb'))
cv = pickle.load(open('count_vectorizer.pkl', 'rb'))
model=tf.keras.models.load_model('my_model.h5')

st.write("""
# Emotion detection through text 
"Enter your text and discover its emotion! Our model is trained on various emotions like sadness, joy, love, anger, fear, and surprise to provide accurate predictions."
""")


def user_input():
    text = st.text_area('Enter your sentence here: ')

    return text
input = user_input()
input=preprocess(input)
array = cv.transform([input]).toarray()
pred = model.predict(array)
a=np.argmax(pred, axis=1)
prediction = encoder.inverse_transform(a)[0]
if input:

 st.markdown(f'<p class="big-font">Your predicted emotion is: <span style="color:#ff5733">{prediction}</span></p>', unsafe_allow_html=True)
else:
 st.markdown(f'<p class="big-font"> Your predicted emotion is: <span style="color:#ff5733"></span></p>', unsafe_allow_html=True)
