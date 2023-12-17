from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import seaborn as sns

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
    # assert all(label for label in labels)
    # cloud_types = ["Ac", "As", "Cb", "Cc", "Ci", "Cs", "Ct", "Cu", "Ns", "Sc", "St"]
    # for label in labels:
    #     assert label in cloud_types, f"{label=} not in {cloud_types=}"
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


def ffnn_16x16_regularization():
    feed_forward_neural_network()


def grid_search():
    """idk"""
    architectures = [
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
        (
            200,
            200,
            200,
        ),
        (200, 200, 200, 200),
        (400),
        (400, 400),
        (
            400,
            400,
            400,
        ),
        (400, 400, 400, 400),
        (800),
        (800, 800),
        (
            800,
            800,
            800,
        ),
        (800, 800, 800, 800),
        (1600,),
        (1600, 1600),
        (
            1600,
            1600,
            1600,
        ),
        (1600, 1600, 1600, 1600),
    ]
    for filename in FILENAMES:
        for architecture in architectures[16:24]:
            (
                train_accuracy,
                test_accuracy,
                elapsed_training_time,
                clf
            ) = feed_forward_neural_network(filename, architecture)
            print(
                f"{train_accuracy=:.4g}, {test_accuracy=:.4g}, {clf.n_iter_=:.4g}, {architecture=}"
            )

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
            cfl
        ) = feed_forward_neural_network(filename, architecture)
        
        # Append the results to the list
        results.append({
            'Train Accuracy': train_accuracy,
            'Test Accuracy': test_accuracy,
            'Iterations': cfl.n_iter_,
            'Architecture': architecture,
            'model':_+1
        })
    # Create a DataFrame from the results list
    df = pd.DataFrame(results)
    # Print the DataFrame
    print(df)
    # Create a bar plot using Seaborn to visualize test accuracies for each architecture
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='model', y='Test Accuracy')
    plt.xticks(rotation=45)
    plt.xlabel('Model')
    plt.ylabel('Test Accuracy')
    plt.tight_layout()

    # Display the plot
    plt.show()



def unveil_possible_overfitting():
    """one particular model trained on 16x16 had very high train accuracy,
    but seemingly capped test accuracy.
    showing history to uncover possible bias variance.
    """
    architecture=(541, 309, 183, 239, 939, 950)
    (
        train_accuracy,
        test_accuracy,
        elapsed_training_time,
        clf
    ) = feed_forward_neural_network(FILENAMES[0], architecture)  
    loss = clf.loss_curve_
    validation = clf.validation_scores_
    iters=np.arange(1, len(loss)+1)
    print(
        f"{train_accuracy=:.4g}, {test_accuracy=:.4g}, {elapsed_training_time=:.4g}, {architecture=}"
    )
    plt.plot(iters, loss, label="loss")
    plt.plot(iters, validation, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    gridsearch_architectures(FILENAMES[2])
    # gridsearch_architecture_32x32_relu()
    # gridsearch_architecture_64x64_relu()
    # gridsearch_architecture_128x128_relu()
    # unveil_possible_overfitting()
    # train_by_input()
