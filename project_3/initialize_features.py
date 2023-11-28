"""parse `CCSN_v2` for all .jpg to store as elements in features
"""
from PIL import Image
import numpy as np
from pathlib import Path


def read_image(filename: str, target_size=(256, 256)) -> np.array:
    """
    Read an image file, resize it to target_size, and return it as an rgb array.

    Parameters
    ----------
    filename : str or Path
        Path to the image file.
    target_size : tuple, optional
        Target size for resizing, default is (256, 256).

    Returns
    -------
    np.ndarray
        Resized image as an rgb array.
    """
    assert isinstance(filename, (str, Path))
    
    # Open the image and resize it to the target size while maintaining the aspect ratio
    img = Image.open(filename)
    img = img.resize(target_size, Image.LANCZOS)
    
    # Convert the resized image to an rgb array
    rgb_array = np.asarray(img)
    
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


def get_label(filename: str) -> str:
    """Grab label from filename
    """
    label = filename.split("-")[0]
    return label


def get_feature_and_label_feed_forward(filename: str) -> tuple:
    """Get feature and label for feed forward neural network.
    """
    label = get_label(filename)
    image_arr = read_image(filename)
    image_data = flatten_image_arr(image_arr)
    return image_data, label



def count_image_sizes():
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
# of images with shape {expected_shape_2}: {expected_shape_2_count}""")
    



def features_and_targets_feed_forward_neural_network():
    """produce features and labels compatible with feed forward neural network.
    """
    dir = Path.cwd()
    if not dir.is_dir():
        raise NotADirectoryError
    
    image_filenames = dir.rglob("**/*.jpg")
    n_images = sum(1 for _ in image_filenames)
    n_image_vals = 256*256*3
    labels = np.empty(n_images, dtype=str)
    features =  np.empty((n_images, n_image_vals), dtype=int)
    for i , image_filename in enumerate(image_filenames):
        features[i, :], labels[i] = get_feature_and_label_feed_forward(image_filename)
    return features, labels
    



if __name__ == "__main__":
    features, labels = features_and_targets_feed_forward_neural_network()