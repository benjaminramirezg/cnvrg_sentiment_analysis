import os
import pickle
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import text
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

parser = argparse.ArgumentParser(description='Training sentiment analysis model')
parser.add_argument('--epochs', help='number of epochs to run', default=2)
parser.add_argument('--batch_size', help='iteration batch size', default=16)
parser.add_argument('--dropout', help='dropout', default=0.2)
parser.add_argument('--learning_rate', help='learning rate', default=0.001)
parser.add_argument('--num_words', help='Bag of Words size', default=10000)
args = parser.parse_args()

DATASET_PATH = '/data/sentimentanalysis/dataset.csv'
MODEL_PATH = './output/'
HYPERPARAMETERS = {
    'NUM_WORDS': int(args.num_words),
    'BATCH_SIZE': int(args.batch_size),
    'EPOCHS': int(args.epochs),
    'DROPOUT': float(args.dropout),
    'LEARNING_RATE': float(args.learning_rate),
}

def train():
    """Trains a model given a dataset"""
    NUM_WORDS = HYPERPARAMETERS['NUM_WORDS']
    BATCH_SIZE = HYPERPARAMETERS['BATCH_SIZE']
    EPOCHS = HYPERPARAMETERS['EPOCHS']

    dataset = pd.read_csv(DATASET_PATH, sep='\t')
    texts = dataset["text"].tolist()
    labels = dataset["label"].tolist()

    tokenizer = text.Tokenizer(num_words=NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    X = tokenizer.texts_to_matrix(texts, mode="tfidf")
    y = np.array(labels)

    model = Sequential()
    model.add(Dense(250, input_shape=(NUM_WORDS,)))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.25, verbose=2)
       
    evaluation = model.evaluate(X, y)
    metrics = {'loss': float(evaluation[0]), 'accuracy': float(evaluation[1])}

    nn_path = '{}/model.h5'.format(MODEL_PATH)
    preprocessor_path = '{}/tokenizer.pkl'.format(MODEL_PATH)
    model.save(nn_path)
    pickle.dump(tokenizer, open(preprocessor_path,'wb+'), protocol=pickle.HIGHEST_PROTOCOL)
    return metrics

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)
    metrics = train()
    for key, value in metrics.items():
        print("cnvrg_tag_{}: {}".format(key, str(value)))