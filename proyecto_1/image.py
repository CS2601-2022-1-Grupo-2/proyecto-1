import numpy as np
import os
import pywt

from PIL import Image

def process_image(path: str, size: int = 64, n: int = 4) -> np.ndarray:
    with Image.open(path).convert("L").resize((size, size)) as img:
        LL = np.array(img.getdata(), dtype=np.float32)/255
        LL = LL.reshape(img.size[0], img.size[1])

        for _ in range(n):
            LL = pywt.dwt2(LL, "haar")[0]

        return LL.reshape(-1)

def get_vectors(path: str, test: bool):
    v = []
    y = []

    if test:
        for characteristic in os.scandir(path):
            for file in os.scandir(characteristic):
                v.append(process_image(file.path))
                y.append(characteristic.name)
    else:
        for file in os.scandir(path):
            v.append(process_image(file.path))
            y.append(None)

    return (np.array(v), y)
