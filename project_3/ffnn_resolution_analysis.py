from PIL import Image
import numpy as np
from pathlib import Path


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
    n_image_vals = image_size[0]*image_size[1]*3
    features =  np.empty((n_images, n_image_vals), dtype=int)
    labels = np.empty(n_images, dtype=f'<U{2}')   
    for i , image_path in enumerate(image_paths):
        feature, label = get_feature_and_label_feed_forward(image_path, image_size=image_size)
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

if __name__ == "__main__":
    resolutions = [16, 32, 64, 128]
    for resolution in resolutions:
        features_and_labels_feed_forward_neural_network(image_size=(resolution, resolution))