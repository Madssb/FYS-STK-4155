"""parse `CCSN_v2` for all .jpg and store each as a feature in a features array.
"""
from PIL import Image
import numpy as np
from pathlib import Path


def read_image(filename: str, target_size: tuple=(28, 28),
               resample: Image=Image.LANCZOS) -> np.ndarray:
    """
    Read an image file, resize it to target_size, and return it as an rgb array.

    Parameters
    ----------
    filename : str or Path
        Path to the image file.
    target_size : tuple, optional
        Target size for resizing, default is (256, 256).
    resample : Image, optional
        Resampling filter.
    
    Returns
    -------
    np.ndarray
        Resized image as an rgb array.
    """
    assert isinstance(filename, (str, Path))
    image = Image.open(filename)
    image = image.resize(target_size, resample=resample)
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


def get_label(filename: str or Path) -> str:
    """Grab label from filename
    """
    filename = str(filename)
    label = filename.split("-")[0]
    return label


def get_feature_and_label_feed_forward(filename: str, target_size: tuple=(28, 28),
                                       resample: Image=Image.LANCZOS) -> tuple:
    """Get feature and label from .jpg for feed forward neural network.
    """
    label = get_label(filename)
    image_arr = read_image(filename, target_size=target_size, resample=resample)
    image_data = flatten_image_arr(image_arr)
    return image_data, label



def image_diagnostics():
    """Display sizes of the jpgs used 
    """
    dir = Path.cwd()
    if not dir.is_dir():
        raise NotADirectoryError
    
    image_filenames = dir.rglob("**/*.jpg")
    expected_shape_1 = (400, 400, 3)
    expected_shape_2 = (256, 256, 3)
    expected_shape_1_count = 0
    expected_shape_2_count = 0
    img_count = 0
    other_shape_count = 0
    for image_filename in image_filenames:
        image_arr = read_image(image_filename)
        if image_arr.shape == expected_shape_1:
            expected_shape_1_count +=1
        elif image_arr.shape == expected_shape_2:
            expected_shape_2_count += 1
        else:
            other_shape_count += 1
        img_count += 1
    print(f"""# of images: {img_count}
# of images with shape {expected_shape_1}: {expected_shape_1_count}
# of images with shape {expected_shape_2}: {expected_shape_2_count}
# of images with other shapes {other_shape_count}""")
    



def features_and_targets_feed_forward_neural_network(image_size=(256, 256)):
    """produce features and labels compatible with feed forward neural network.
    """
    dir = Path.cwd()
    if not dir.is_dir():
        raise NotADirectoryError
    image_filenames = list(dir.rglob("**/*.jpg"))
    n_images = sum(1 for _ in image_filenames)
    n_image_vals = 256*256*3
    labels = np.empty(n_images, dtype=str)
    features =  np.empty((n_images, n_image_vals), dtype=int)
    for i , image_filename in enumerate(image_filenames):
        print(i)
        features[i, :], labels[i] = get_feature_and_label_feed_forward(image_filename)
    return features, labels


if __name__ == "__main__":
    dir = Path.cwd()
    if not ((dir/"features.npy").is_file() and (dir/"labels.npy").is_file()):
        features, labels = features_and_targets_feed_forward_neural_network()
        np.save("features.npy", features)
        np.save("labels.npy", labels)
    features = np.load("features.npy")
    labels = np.load("labels.npy")
    print(features.shape)
    print(labels.shape) 
