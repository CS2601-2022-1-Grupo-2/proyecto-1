import numpy as np
import pywt

from PIL import Image

def process_image(path: str, size: int = 64) -> np.ndarray:
    with Image.open(path).convert("L").resize((size, size)) as img:
        LL = np.array(img.getdata(), dtype=np.float32)/255
        LL = LL.reshape(img.size[0], img.size[1])

        # TODO apply haar

        return LL.reshape(-1)
