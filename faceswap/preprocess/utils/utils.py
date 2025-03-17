import cv2
import numpy as np


def resize_to_limit(img: np.ndarray, max_dim=1280, division=2):
    h, w = img.shape[:2]

    if max_dim > 0 and max(h, w) > max_dim:
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        img = cv2.resize(img, (new_w, new_h))

    division = max(division, 1)
    new_h = img.shape[0] - (img.shape[0] % division)
    new_w = img.shape[1] - (img.shape[1] % division)

    if new_h == 0 or new_w == 0:
        return img

    if new_h != img.shape[0] or new_w != img.shape[1]:
        img = img[:new_h, :new_w]

    return img
