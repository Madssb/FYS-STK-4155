"""Feed Forward Neural Network training on image data that has been resized to 258x258, and otherwise unprocessed
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import datetime
import pytz
import json


def something():
    norway_timezone = pytz.timezone('Europe/Oslo')
    architechtures = [
        (50),
        (50, 50),
        (50, 50, 50),
        (50, 50, 50, 50),
        (100),
        (100, 100),
        (100, 100, 100),
        (100, 100, 100, 100),
        (200),
        (200, 200),
        (200, 200, 200,),
        (200, 200, 200, 200),
        (200),
        (400, 400),
        (400, 400, 400,),
        (400, 400, 400, 400),
        (800),
        (800, 800),
        (800, 800, 800,),
        (800, 800, 800, 800),
    ]
    scores = np.empty(len(architechtures), dtype=float)


    size = "128x128"
    features = np.load(f"features_{size}.npy")
    labels = np.load(f"labels_{size}.npy")
    assert all(label for label in labels)
    cloud_types = ["Ac", "As", "Cb", "Cc", "Ci", "Cs", "Ct", "Cu", "Ns", "Sc", "St"]
    for label in labels:
        assert label in cloud_types, f"{label=} not in {cloud_types=}"
    print("made it past assertion")

    onehot_encoder = OneHotEncoder()
    onehot_labels =  onehot_encoder.fit_transform(np.array(labels).reshape(-1, 1))

    features_train, features_test, labels_train, labels_test = train_test_split(features, onehot_labels, random_state=1, shuffle=True)
    for i, architechture in enumerate(architechtures):
        print(f"hidden layer config: {architechture}")
        start_train_time_norway = datetime.datetime.now(norway_timezone)
        print("starting training ",start_train_time_norway)
        clf = MLPClassifier(random_state=1, hidden_layer_sizes=architechture)
        clf.fit(features_train, labels_train)
        finish_train_time_norway = datetime.datetime.now(norway_timezone)
        print("finished training ", finish_train_time_norway)
        # model_params = clf.get_params()
        # with open(f'model_weights_ffnn_{size}.json', 'w') as json_file:
        #     json.dump(model_params, json_file, indent=4)
        score_trained  = clf.score(features_test, labels_test)
        print("trained sore:",score_trained)
        scores[i] = score_trained
    # print("improvement:", score_trained - score_untrained)  




def train_256():
    norway_timezone = pytz.timezone('Europe/Oslo')
    features = np.load("128/features_ffnn.npy")
    labels = np.load("labels.npy")
    print("here")
    # eh = features.shape[0]//32
    # features = features[eh:, :]
    # labels = labels[eh:]
    onehot_encoder = OneHotEncoder()
    onehot_labels =  onehot_encoder.fit_transform(np.array(labels).reshape(-1, 1))
    print("pre split")
    input()
    features_train, features_test, labels_train, labels_test = train_test_split(features, onehot_labels, random_state=1, shuffle=True)
    input()
    print("here 2")
    start_train_time_norway = datetime.datetime.now(norway_timezone)
    print("starting training ",start_train_time_norway)
    clf = MLPClassifier(random_state=1, hidden_layer_sizes=(100,))
    clf.fit(features_train, labels_train)
    finish_train_time_norway = datetime.datetime.now(norway_timezone)
    print("finished training ", finish_train_time_norway)
    model_params = clf.get_params()
    with open(f'model_weights_ffnn_256x256_1h_100n.json', 'w') as json_file:
        json.dump(model_params, json_file, indent=4)
    score_trained  = clf.score(features_test, labels_test)
    print("trained sore:",score_trained)

if __name__ == "__main__":
    train_256()