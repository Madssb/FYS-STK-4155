"""Analysis of how various image resampling resolutions affect
the trained model for FFNN for classification of cloud images.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import seaborn as sns
from PIL import Image
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from utilities import my_figsize

FILENAMES = [
    "features_ffnn_16x16.npy",
    "features_ffnn_32x32.npy",
    "features_ffnn_64x64.npy",
    "features_ffnn_128x128.npy",
]


def read_image(
    jpg_path: Path, image_size: tuple = (256, 256), resample: Image = Image.LANCZOS
) -> np.ndarray:
    """
    Read an image file, resize it to image_size, and return it as an rgb array.

    Parameters
    ----------
    jpg_path : Path
        Path to the image file.
    image_size : tuple, optional
        Target size for resizing, default is (256, 256).
    resample : Image, optional
        Resampling filter.

    Returns
    -------
    np.ndarray
        Resized image as an rgb array.
    """
    assert isinstance(jpg_path, (str, Path))
    image = Image.open(jpg_path)
    image = image.resize(image_size, resample=resample)
    rgb_array = np.array(image, dtype=int)
    return rgb_array


def flatten_image_arr(rgb_array: np.ndarray) -> np.ndarray:
    """Flatten rgb image in array form to flat array form.

    Parameters
    ----------
    rgb_array : np.ndarray
        rgb_array to flatten

    Returns
    -------
    np.ndarray
        flattened rgb array.
    """
    return rgb_array.flatten()


def get_label(jpg_path: Path) -> str:
    """Grab label from path to .jpg in CCSN_v2

    Parameters
    ----------
    jpg_path : Path
        Path to .jpg in CCSN_v2 data.

    Returns
    -------
    str
        Image label
    """
    jpg_path = Path(jpg_path)
    if not jpg_path.suffix == ".jpg":
        raise TypeError
    label = jpg_path.parent.name
    return label


def get_feature_and_label_feed_forward(
    jpg_path: str, image_size: tuple = (256, 256), resample: Image = Image.LANCZOS
) -> tuple:
    """Get feature and label from .jpg for feed forward neural network.

    Parameters
    ----------
    jpg_path : Path
        Path to .jpg in CCSN_v2 data.

    """
    label = get_label(jpg_path)
    image_arr = read_image(jpg_path, image_size=image_size, resample=resample)
    flat_image_arr = flatten_image_arr(image_arr)
    return flat_image_arr, label


def features_and_labels_feed_forward_neural_network(image_size=(128, 128)):
    """Produce features and labels compatible with feed forward neural network, given that they dont already exist.

    Parameters
    ----------
    image_size : tuple
    """
    dir = Path.cwd()
    image_paths = sorted(dir.rglob("**/*.jpg"))
    n_images = len(image_paths)
    n_image_vals = image_size[0] * image_size[1] * 3
    features = np.empty((n_images, n_image_vals), dtype=int)
    labels = np.empty(n_images, dtype=f"<U{2}")
    for i, image_path in enumerate(image_paths):
        feature, label = get_feature_and_label_feed_forward(
            image_path, image_size=image_size
        )
        features[i, :] = feature
        labels[i] = label
        print(f"{i}/{n_images}")
    size_str = f"{image_size[0]}x{image_size[1]}"
    features_filename = Path(f"features_ffnn_{size_str}.npy")
    labels_filename = Path(f"labels.npy")
    if not features_filename.is_file():
        np.save(features_filename, features)
    if not labels_filename.is_file():
        np.save(labels_filename, labels)


def initialize_features():
    """Initialize features"""
    resolutions = [16, 32, 64, 128]
    for resolution in resolutions:
        features_and_labels_feed_forward_neural_network(
            image_size=(resolution, resolution)
        )


def feed_forward_neural_network(
    feature_filename: str,
    architecture: tuple[int],
    learning_rate_init: float = 0.001,
    activation: str = "relu",
):
    """Assemble feed forward neural network and train with ADAM.

    Parameters
    ----------
    feature_filename : str or Path
        Filename for features stored as .npy.
    architecture: Tuple[int].
        architecture of ffnn to be trained
    learning_rate_init : float
        Initial learning rate.


    Returns
    -------

    """
    labels = np.load("labels.npy")
    features = np.load(feature_filename)
    features = (features - np.mean(features, axis=0) / np.std(features, axis=0))
    assert all(label for label in labels)
    cloud_types = ["Ac", "As", "Cb", "Cc", "Ci", "Cs", "Ct", "Cu", "Ns", "Sc", "St"]
    for label in labels:
        assert label in cloud_types, f"{label=} not in {cloud_types=}"
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels)
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, integer_labels, random_state=1, shuffle=True
    )
    clf = MLPClassifier(
        random_state=1,
        hidden_layer_sizes=architecture,
        learning_rate_init=learning_rate_init,
        activation=activation,
        early_stopping=True
    )
    time_pre_training = time.time()
    clf.fit(features_train, labels_train)
    elapsed_training_time = time.time() - time_pre_training
    train_accuracy = clf.score(features_train, labels_train)
    test_accuracy = clf.score(features_test, labels_test)
    return train_accuracy, test_accuracy, elapsed_training_time, clf


def plot_confusion_matrix():
    """Visualize the confusion matrices for the test set for
    the best performing models
    """
    sizes = ["16x16", "32x32", "64x64", "128x128"]
    for i, size in enumerate(sizes):
        labels = np.load("labels.npy")
        features = np.load(FILENAMES[i])
        features = (features - np.mean(features, axis=0) / np.std(features, axis=0))
        label_encoder = LabelEncoder()
        integer_labels = label_encoder.fit_transform(labels)
        features_train, features_test, labels_train, labels_test = train_test_split(
            features, integer_labels, random_state=1, shuffle=True
        )
        clf = feed_forward_neural_network(FILENAMES[i], (39, 90, 69, 92, 68, 39))[3]
        # Predict labels for the test dataset
        predicted_labels = clf.predict(features_test)

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(labels_test, predicted_labels, normalize="true")

        print("Confusion Matrix:")
        print(conf_matrix)
        plt.figure(figsize=my_figsize(column=False))
        sns.set(font_scale=1.0)  # Adjust font size for readability
        sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", 
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.savefig(f"confusion_{size}.pdf")


def gridsearch_architectures(filename):
    """Search for the best architecture for fixed image resolution and activation."""
    random.seed(2023)
    
    def random_tuple():
        """Generate a random architecture tuple."""
        temp = []
        for _ in range(random.randint(1, 6)):
            temp.append(random.randint(1, 100))
        return tuple(temp)

    architectures = []
    results = []

    for _ in range(20):
        architecture = random_tuple()
        architectures.append(architecture)
        (
            train_accuracy,
            test_accuracy,
            elapsed_training_time,
            clf
        ) = feed_forward_neural_network(filename, architecture)
        
        # Append the results to the list
        results.append({
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy,
            'Iterations': clf.n_iter_,
            'Architecture': architecture,
            'Model':_+1,
            'depth':len(architecture)
        })
    # Create a DataFrame from the results list
    df = pd.DataFrame(results)
    df = df.sort_values(by='Test Accuracy', ascending=False)
    return df


def size_analysis():
    """Visualize accuracy scores for all models, for each image resolution.
    """
    sizes = ["16x16", "32x32", "64x64", "128x128"]
    for i, size in enumerate(sizes):
    #for i, size in enumerate(sizes[:-1]):
        df = gridsearch_architectures(FILENAMES[i])
        plt.figure(figsize=my_figsize(column=False))
        bar_width = 0.4  # Width of each bar
        index = np.arange(len(df))  # Index for x-axis
        plt.bar(index, df['Train Accuracy'], bar_width, label='Train Accuracy', color='steelblue')
        plt.bar(index + bar_width, df['Test Accuracy'], bar_width, label='Test Accuracy', color='coral')
        plt.xticks(index + bar_width / 2, df['Model'])
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"accuracies_{size}.pdf")


def size_analysis_128():
    df = gridsearch_architectures(FILENAMES[3])
    plt.figure(figsize=my_figsize(column=False))
    bar_width = 0.4  # Width of each bar
    index = np.arange(len(df))  # Index for x-axis
    plt.bar(index, df['Train Accuracy'], bar_width, label='Train Accuracy', color='steelblue')
    plt.bar(index + bar_width, df['Test Accuracy'], bar_width, label='Test Accuracy', color='coral')
    plt.xticks(index + bar_width / 2, df['Model'])
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"accuracies_128x128.pdf")    


def best_model_analysis():
    """show how loss curves for best models evolve.
    """
    sizes = ["16x16", "32x32", "64x64", "128x128"]
    for i, size in enumerate(sizes):
        if i==3:
            clf = feed_forward_neural_network(FILENAMES[i], (95, 79))[3]
        else:
            clf = feed_forward_neural_network(FILENAMES[i], (39, 90, 69, 92, 68, 39))[3]
        loss = clf.loss_curve_
        iters=np.arange(1, len(loss)+1)
        plt.plot(iters, loss, label=size)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.savefig(f"loss.pdf")


def unveil_possible_overfitting():
    """one particular model trained on 16x16 had very high train accuracy,
    but seemingly capped test accuracy.
    showing history to uncover possible bias variance.
    """
    architecture=(541, 309, 183, 239, 939, 950)
    clf = feed_forward_neural_network(FILENAMES[0], architecture)[3]
    loss = clf.loss_curve_
    validation = clf.validation_scores_
    iters=np.arange(1, len(loss)+1)
    plt.plot(iters, loss, label="loss")
    plt.plot(iters, validation, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # plot_confusion_matrix()
    # unveil_possible_overfitting()
    # size_analysis()
    # best_model_analysis()
