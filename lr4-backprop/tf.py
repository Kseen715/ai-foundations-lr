import os

from backprop import *

from tqdm.keras import TqdmCallback
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np


def load_model(model_filename: str):
    '''
    Loads the model from model.keras.
    '''
    global model
    try:
        model = keras.models.load_model(model_filename)
        print("Model " + model_filename + " loaded.")
        return model
    except Exception as e:
        print("Failed to load model: " + str(e))


def predict(predict_pd: pd.DataFrame, verbose: int = 1):
    '''
    Predicts the result based on config.
    '''
    global model
    prediction = model.predict(predict_pd, verbose=verbose)[0]
    max_prediction = max(prediction)
    # prediction_id = np.where(prediction, max(prediction))[0] + 1
    prediction_id = np.where(np.isclose(prediction, max(prediction)))[0] + 1
    # if prediction_id is list:
    prediction_id = prediction_id[0]
    return prediction, max_prediction, prediction_id


def tf_train(epoch_count: int = 1000):
    global samples, strings, model
    train_data = samples[:-1]
    train_labels = samples[-1]
    # lb_logger.log_info("Train data: " + str(train_data))
    # lb_logger.log_info("Train labels: " + str(train_labels))

    model = keras.Sequential([
        keras.layers.Dense(1024, activation='sigmoid'),
        keras.layers.Dense(768, activation='sigmoid'),
        keras.layers.Dense(6, activation='sigmoid'),
        # keras.layers.Dense(6)
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])

    verbosity = 1

    model.fit(train_data, train_labels, epochs=epoch_count,
              verbose=0, callbacks=[TqdmCallback(verbose=verbosity)])

    test_loss, test_acc = model.evaluate(train_data,  train_labels, verbose=0)

    print('Test accuracy: {}'.format(test_acc))
    # print('Test loss: {}'.format(test_loss))

    # save model
    max_model_id = 0
    for filename in os.listdir('models'):
        if filename.startswith('model_') and filename.endswith('.keras'):
            try:
                model_id = int(filename[6:-6])
                if model_id > max_model_id:
                    max_model_id = model_id
            except:
                pass

    model_id = max_model_id + 1
    four_digit_model_id = str(model_id).zfill(4)
    model_filename = 'models/model_' + four_digit_model_id + '.keras'
    model.save(model_filename)
    print('Model saved as ' + model_filename)


def tf_main():
    global input_neurons, hidden_neurons, output_neurons, samples, strings, LEARN_RATE, actual, hidden

    # Constants
    # hidden_neurons = 768
    # output_neurons = 3
    # LEARN_RATE = 0.00125

    sides = 32
    count = 10000
    input_neurons = sides * sides
    print(f"Input neurons: {input_neurons}")
    print(f"Hidden neurons: {hidden_neurons}")
    print(f"Output neurons: {output_neurons}")
    print(f"Learn rate: {LEARN_RATE}")
    read_json_names(f"data/names_{sides}x{sides}_{count}.json")
    # samples [
    #   [0, 0, 0, [0, 0, 0]],
    # ]
    read_zip_to_data_array(f"data/processed_{sides}x{sides}_{count}.zip") 


    tf_train(1000)


if __name__ == "__main__":
    tf_main()
