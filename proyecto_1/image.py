import numpy as np
import pywt

from PIL import Image

def process_image(path: str) -> np.ndarray:
    with Image.open(path).convert("L") as img:
        LL = np.array(img.getdata(), dtype=np.float32)/255
        LL = LL.reshape(img.size[0], img.size[1])

        # TODO resize image and apply haar

        return LL.reshape(-1)
