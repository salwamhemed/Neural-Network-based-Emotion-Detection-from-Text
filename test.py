from preprocess import preprocess
import pickle
import numpy as np
import tensorflow as tf 

encoder = pickle.load(open('encoder.pkl', 'rb'))
cv = pickle.load(open('count_vectorizer.pkl', 'rb'))
model=tf.keras.models.load_model('my_model.h5')
text= "I am surprised "
array = cv.transform([text]).toarray()
pred = model.predict(array)
a=np.argmax(pred, axis=1)
print(encoder.inverse_transform(a)[0])