import os
import io
import numpy as np
from typing import *
import base64
from pathlib import Path

import cv2
from PIL import Image

def load_image(img: Union[str, np.ndarray]) -> Tuple[np.ndarray, str]:
    """
    args:
        img: path || url || base64 || numpy array
    returns:
        image: loaded img in BRG format (numpy array)
        image_name: str
    """
    
    if isinstance(img, np.ndarray):
        return img

    if isinstance(img, Path):
        img = str(img)
    
    if not isinstance(img, str):
        raise ValueError(f"img must be numpy array or str but received {type(img)}")

    if img.startswith("data:image/"):
        return load_image_from_base64(img), "base64 encoded string"

    if not os.path.isfile(img):
        raise ValueError(f"img is not existed")

    img_obj_bgr = cv2.imread(img)
    return img_obj_bgr

def load_image_from_base64(uri: str) -> np.ndarray:
    """
    args:
        uri: base64 string
    returns:
        numpy array: loaded img
    """
    encoded_data_parts = uri.split(",")
    
    if len(encoded_data_parts) < 2:
        raise ValueError(f"format error in base64 encoded string")

    encoded_data = encoded_data_parts[1]
    decoded_bytes = base64.b64decode(encoded_data)

    with Image.open(io.BytesIO(decoded_bytes)) as img:
        file_type = img.format.lower()
        if file_type not in {"jpeg", "png"}:
            raise ValueError(f"input image can be jpg or png, but it is {file_type}")

    nparr = np.frombuffer(decoded_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_bgr