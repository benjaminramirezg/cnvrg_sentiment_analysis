import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

THRESHOLD = 0.5
MODEL_PATH = './output/model.h5'
PREPROCESSOR_PATH = './output/tokenizer.pkl'
model = tf.keras.models.load_model(MODEL_PATH)
preprocessor = None
with open(PREPROCESSOR_PATH, 'rb') as fh:
    preprocessor = pickle.load(fh)

def predict(text):
    X = preprocessor.texts_to_matrix([text], mode='tfidf')
    p = model.predict(X)
    p = [1 if x[0] > THRESHOLD else 0 for x in p]
    return p