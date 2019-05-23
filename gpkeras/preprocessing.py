from typing import Tuple
import numpy as np


def crop_array_center(arr, dims: Tuple[int, ...]):
    arr_height, arr_width = arr.shape[0], arr.shape[1]
    top = (arr_height - dims[0]) // 2
    left = (arr_width - dims[1]) // 2
    return arr[top:top+dims[0], left:left+dims[1]]


def change_labels(arr: np.ndarray, new_labels: Tuple[Tuple[int, ...], ...]):
    arr2 = np.zeros(arr.shape, dtype=np.uint8)
    for i in range(len(new_labels)):
        for l in new_labels[i]:
            arr2[arr == l] = i
    return arr2
