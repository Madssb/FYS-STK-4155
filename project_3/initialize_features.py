"""parse `CCSN_v2` for all .jpg to store as elements in features
"""
from PIL import Image
import numpy as np


def read_image(filename: str) -> np.array:
    """Read an image file to an rgb array.
    """
    return np.asarray(Image.open(filename))


def flatten_image_arr(rgb_array: np.ndarray) -> np.ndarray:
    """Flatten rgb image in array form to flat array form.
    """
    return rgb_array.flatten()


def temp(filename):
    label = 




if __name__ == "__main__":
    file_path = "CCSN_v2/Ac/Ac-N001.jpg"
    rgb_arr = read_image(file_path)
    print(rgb_arr.shape)
    print(flatten_image_arr(rgb_arr).shape)