import numpy as np
from PIL import Image

def get_img(img_path="../images/cameraman.tif"):
    img = np.array(Image.open(img_path)).astype('float64') / 255
    return img


def save_img(img, output_name="output_img.png"):
    if img.shape[-1] == 3:
        Image.fromarray((abs(img) * 255).astype('uint8'), 'RGB').save(output_name)
    else:
        Image.fromarray((abs(img) * 255).astype('uint8')).save(output_name, optimize=True, compress_level=0)

