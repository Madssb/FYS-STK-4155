"""parse `CCSN_v2` for all .jpg and store each as a feature in a features array.


TO DO:
implement extending the dataset by rotating images and retaining label.
"""
from PIL import Image
import numpy as np
from pathlib import Path
from PIL import ImageEnhance

def read_image(jpg_path: Path, image_size: tuple=(256, 256),
               resample: Image=Image.LANCZOS,
               increase_contrast=False) -> np.ndarray:
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
    if increase_contrast:
        enhancer = ImageEnhance.Contrast(image)
        image.enhance(10)
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


def get_feature_and_label_feed_forward(jpg_path: str, image_size: tuple=(256, 256),
                                       resample: Image=Image.LANCZOS,
                                       increase_contrast=False) -> tuple:
    """Get feature and label from .jpg for feed forward neural network.

    Parameters
    ----------
    jpg_path : Path
        Path to .jpg in CCSN_v2 data.
    
    """
    label = get_label(jpg_path)
    image_arr = read_image(jpg_path, image_size=image_size, resample=resample, increase_contrast=increase_contrast)
    flat_image_arr = flatten_image_arr(image_arr)
    return flat_image_arr, label




def get_feature_and_label_convolutional(jpg_path: str, image_size: tuple=(256, 256),
                                        resample: Image=Image.LANCZOS,
                                        increase_contrast=False) -> tuple:
    """Get feature and label from .jpg for Convolutional neural network.
    """
    label = get_label(jpg_path)
    image_arr = read_image(jpg_path)
    return image_arr

def diagnostics():
    """Display sizes of the jpgs used 
    """
    dir = Path.cwd()
    if not dir.is_dir():
        raise NotADirectoryError
    
    image_paths = dir.rglob("**/*.jpg")
    expected_shape_1 = (400, 400, 3)
    expected_shape_2 = (256, 256, 3)
    expected_shape_1_count = 0
    expected_shape_2_count = 0
    img_count = 0
    other_shape_count = 0
    for image_path in image_paths:
        image_arr = read_image(image_path)
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
    



def features_and_labels_feed_forward_neural_network(image_size=(128, 128)):
    """Produce features and labels compatible with feed forward neural network, given that they dont already exist.

    Parameters
    ----------
    image_size : tuple
    """
    assert image_size[0] == image_size[1], "image size must be square"
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
    output_dir = dir / size_str
    if not output_dir.is_dir():
        output_dir.mkdir()
    features_filename = Path(output_dir / f"features_ffnn.npy")
    labels_filename = Path(f"labels.npy")
    if not features_filename.is_file():
        np.save(features_filename, features)
    if not labels_filename.is_file():
        np.save(labels_filename, labels)

def features_and_labels_feed_forward_neural_network_high_contrast(image_size=(128, 128)):
    """Produce features and labels compatible with feed forward neural network, given that they dont already exist.

    Parameters
    ----------
    image_size : tuple
    """
    assert image_size[0] == image_size[1], "image size must be square"
    dir = Path.cwd()
    image_paths = sorted(dir.rglob("**/*.jpg"))
    n_images = len(image_paths)
    n_image_vals = image_size[0]*image_size[1]*3
    features =  np.empty((n_images, n_image_vals), dtype=int)
    labels = np.empty(n_images, dtype=f'<U{2}')   
    for i , image_path in enumerate(image_paths):
        feature, label = get_feature_and_label_feed_forward(image_path, image_size=image_size, increase_contrast=True)
        features[i, :] = feature
        labels[i] = label
        print(f"{i}/{n_images}")
    size_str = f"{image_size[0]}x{image_size[1]}"
    output_dir = dir / size_str
    if not output_dir.is_dir():
        output_dir.mkdir()
    features_filename = Path(output_dir / f"features_ffnn_high_contrast.npy")
    labels_filename = Path(f"labels.npy")
    if not features_filename.is_file():
        np.save(features_filename, features)
    if not labels_filename.is_file():
        np.save(labels_filename, labels)



def features_and_labels_convolutional_neural_network(image_size=(128, 128)):
    """Produce features and labels compatible with feed forward neural network, given that they dont already exist.

    Parameters
    ----------
    image_size : tuple
    """
    assert image_size[0] == image_size[1], "image size must be square"
    dir = Path.cwd()
    image_paths = sorted(dir.rglob("**/*.jpg"))
    n_images = len(image_paths)
    images = np.empty((n_images, image_size[0], image_size[1], 3))

if __name__ == "__main__":
    features_and_labels_feed_forward_neural_network(image_size=(256, 256))



def test_get_label():
    dir = Path.cwd()
    image_paths = list(dir.rglob("**/*.jpg"))
    image_path = image_paths[0]
    expected_label = "Ci"
    label = get_label(image_path)
    assert expected_label == label