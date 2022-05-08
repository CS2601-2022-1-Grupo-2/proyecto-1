import numpy as np
import pywt

from PIL import Image

def process_image(path, size = 64, n = 4) -> np.ndarray:
    with Image.open(path).convert("L").resize((size, size)) as img:
        LL = np.array(img.getdata(), dtype=np.float32)/255
        LL = LL.reshape(img.size[0], img.size[1])

        for _ in range(n):
            LL = pywt.dwt2(LL, "haar")[0]

        return LL.reshape(-1)
