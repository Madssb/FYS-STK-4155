from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import datetime
import pytz
import json
norway_timezone = pytz.timezone('Europe/Oslo')



features = np.load("features_256x256.npy")
labels = np.load("labels.npy")
labels = np.array([label.strip() for label in labels])
assert all(label for label in labels)
cloud_types = ["Ac", "As", "Cb", "Cc", "Ci", "Cs", "Ct", "Cu", "Ns", "Sc", "St"]
invalid_labels = []

for label in labels:
    if label not in cloud_types:
        invalid_labels.append(label)

if invalid_labels:
    # Limit the number of invalid labels displayed
    max_display = 10  # Change this value to display more or fewer labels
    invalid_labels_display = invalid_labels[:max_display]
    num_invalid_labels = len(invalid_labels)

    if num_invalid_labels <= max_display:
        error_message = f"All {num_invalid_labels} labels are not valid cloud types: {', '.join(invalid_labels_display)}"
    else:
        error_message = f"{num_invalid_labels} labels are not valid cloud types. First {max_display} invalid labels: {', '.join(invalid_labels_display)}"
    
    assert False, error_message

onehot_encoder = OneHotEncoder(sparse=False)
onehot_labels =  onehot_encoder.fit_transform(np.array(labels).reshape(-1, 1))

features_train, features_test, labels_train, labels_test = train_test_split(features, onehot_labels, random_state=1, shuffle=True)
start_train_time_norway = datetime.datetime.now(norway_timezone)
print("starting training ",start_train_time_norway)
clf = MLPClassifier(random_state=1, )
score_untrained = clf.score(features_test, labels_test)
print("untrained score:",score_untrained)
clf.fit(features_train, labels_train)
finish_train_time_norway = datetime.datetime.now(norway_timezone)
print("finished training ", finish_train_time_norway)


model_params = clf.get_params()
with open('model_weights_ffnn_256x256.json', 'w') as json_file:
    json.dump(model_params, json_file, indent=4)
score_trained  = clf.score(features_test, labels_test)
print("trained sore:",score_trained)
print("improvement:", score_trained - score_untrained)  