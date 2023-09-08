import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def gradx(input_img):
    """
    Computes the first gradient coordinate
    :param input_img: input grey scaled image. Should be a numpy array.
    :return: derivative in the x direction
    """
    grad_x = np.zeros_like(input_img)
    grad_x[:-1, :] = input_img[1:, :] - input_img[:-1, :]
    return grad_x


def grady(input_img):
    """
    Computes the second gradient coordinate
    :param input_img: input grey scaled image. Should be a numpy array.
    :return: derivative in the y direction
    """
    grad_y = np.zeros_like(input_img)
    grad_y[:, :-1] = input_img[:, 1:] - input_img[:, :-1]
    return grad_y


def grad_operator():
    def derivative_operation(input_array):
        return np.dstack([gradx(input_array), grady(input_array)])

    return derivative_operation


def grad_operation(input_array):
    return np.dstack([gradx(input_array), grady(input_array)])


def _div(px, py):
    """
    Computes the divergence of the vector (px,py).
    px and py should be of the same shape and the result should be such that div = -(grad)^*
    """
    if px.shape != py.shape:
        raise ValueError("input arrays px and py are not of the same shape !")
    div_x = np.zeros_like(px)
    div_y = np.zeros_like(py)

    div_x[1:-1, :] = px[1:-1, :] - px[:-2, :]
    div_x[0, :] = px[0, :]
    div_x[-1, :] = -px[-2, :]

    div_y[:, 1:-1] = py[:, 1:-1] - py[:, :-2]
    div_y[:, 0] = py[:, 0]
    div_y[:, -1] = -py[:, -2]

    return div_x + div_y


def div(input_array):
    """
    Computes the divergence of the vector (px,py).
    px and py should be of the same shape and the result should be such that div = -(grad)^*
    """
    if input_array.shape[-1] != 2:
        raise ValueError("Last dimension should be 2 !")
    px = input_array[..., 0]
    py = input_array[..., 1]

    return _div(px, py)


def laplacian(input_img):
    """
    Computes the laplacian of the input image
    :param input_img: input grey-scaled image.
    :return: the laplacian of the image.
    """
    lp = np.zeros_like(input_img)
    lp[1:-1, 1:-1] = input_img[2:, 1:-1] + input_img[:-2, 1:-1] + input_img[1:-1, 2:] + input_img[1:-1,
                                                                                        :-2] - 4 * input_img[1:-1, 1:-1]
    return lp


def clip_image(input_img, method="cut"):
    """
    Clips the input image values in the interval [0,1].
    Because a numpy array is a mutable object in python, we must copy it beforehand in order to not modify the
       original. In most cases this is unnecessary and suboptimal in terms of memory, but at least it avoids
       unwanted errors popping up from shared use of memory.
    """
    output = input_img.copy()
    if method in ["cut", "normalize"]:
        if method == "cut":
            output[output > 1] = 1
            output[output < 0] = 0
        elif method == "normalize":
            output = output - np.min(output)
            if np.max(output) > 0:
                output = output / np.max(output)
    else:
        raise ValueError(f"incorrect method choice. Available choices : {['«cut»', '«normalize»']}")
    return output


def add_noise(input_img, std):
    """
    Add a Gaussian noise to the input image, with a standard deviation σ.
    :param input_img:
    :param std: standard deviation
    :return: image with an added noise
    """
    rng = np.random.default_rng()
    noise = rng.normal(loc=0, scale=std, size=input_img.shape)
    output = clip_image(input_img + noise)
    return output


def l1_norm(input_img):
    if len(input_img.shape) == 3 and input_img.shape[-1] > 1:
        return np.sum(np.sqrt(np.sum(input_img ** 2, axis=-1)))
    else:
        return np.sum(np.abs(input_img))


def l2_norm(input_img):
    """
    Computes the l2 norm of the input image (a numpy array).
    :param input_img:
    :return:
    """
    return np.sqrt((input_img ** 2).sum())


def change_sign(input_array):
    """
    Detects locations in the input 2d array where there are changes of sign.
    :param input_array:
    :return:
    """
    # second method: via vectorisation and bool multiplication
    bool_matrix = np.full(input_array.shape, False)
    bool_matrix[:-1, :] |= (input_array[:-1, :] * input_array[1:, :] <= 0) & (
            abs(input_array[:-1, :]) <= abs(input_array[1:, :]))
    bool_matrix[1:, :] |= (input_array[:-1, :] * input_array[1:, :] <= 0) & (
            abs(input_array[:-1, :]) > abs(input_array[1:, :]))
    bool_matrix[:, :-1] |= (input_array[:, :-1] * input_array[:, 1:] <= 0) & (
            abs(input_array[:, :-1]) <= abs(input_array[:, 1:]))
    bool_matrix[:, 1:] |= (input_array[:, :-1] * input_array[:, 1:] <= 0) & (
            abs(input_array[:, :-1]) > abs(input_array[:, 1:]))

    # the output is a boolean matrix.
    return bool_matrix.astype('int')


def gaussian_kernel(std, size=(11, 11)):
    """
    Returns the 2-dimensional Gaussian kernel.
    Recall that the formula of the Gaussian kernel is given by:
        G_σ(x) = (1/2πσ²)exp(-|x|²/2σ²)

    :param std: standard deviation of the Gaussian kernel, σ in the formula
    :param size: size of the output array
    :return: The array of the Gaussian kernel
    """
    # Method 1: with vectorisation
    if not (isinstance(size, tuple) and len(size) == 2):
        raise ValueError("Input size should be a 2d tuple of the type (m,n)")
    n, m = size
    grid_i = np.array([np.linspace(-(n // 2), n // 2, n) for _ in range(m)]).T
    grid_j = np.array([np.linspace(-(m // 2), m // 2, m) for _ in range(n)])

    output = np.exp(-(grid_i ** 2 + grid_j ** 2) / (2 * std ** 2)) * (1 / (2 * np.pi * std ** 2))

    # Method 2: by direct input.
    # Much simpler but very slow (around 10 times slower than the previous method).
    # Can be easily optimized by adding the numba decorator @jit.
    # output = np.zeros(shape=size)
    # for i, x in enumerate(np.linspace(-(n//2), n//2, n)):
    #     for j, y in enumerate(np.linspace(-(m//2), (m//2), m)):
    #         output[i, j] = np.exp(-(x**2+y**2)/(2*std**2))*(1/(2*np.pi*std**2))
    return output


def exponent_string(n: int) -> str:
    if not (isinstance(n, int)):
        raise ValueError("The input should be a integer")
    lst = []
    for number in str(n):
        if number == "1":
            lst.append(chr(185))
        elif number in ["2", "3"]:
            lst.append(chr(176 + int(number)))
        else:
            lst.append(chr(8304 + int(number)))
    return "".join(lst)


def get_img(img_path="../images/cameraman.tif"):
    img = np.array(Image.open(img_path)).astype('float64') / 255
    return img


def save_img(img, output_name="output_img.png"):
    if img.shape[-1] == 3:
        Image.fromarray((abs(img) * 255).astype('uint8'), 'RGB').save(output_name)
    else:
        Image.fromarray((abs(img) * 255).astype('uint8')).save(output_name, optimize=True, compress_level=0)


def showimg(image_1, color=True):
    if color is True:
        plt.imshow(image_1)
    else:
        plt.imshow(image_1, cmap='gray')
    plt.show()


def showimg_2(image_1, image_2, color=True):
    if color is True:
        plt.subplot(1, 2, 1)
        plt.imshow((image_1 * 255).astype('uint8'))
        plt.subplot(1, 2, 2)
        plt.imshow((image_2 * 255).astype('uint8'))
    else:
        plt.subplot(1, 2, 1)
        plt.imshow((image_1 * 255).astype('uint8'), cmap="gray")
        plt.subplot(1, 2, 2)
        plt.imshow((image_2 * 255).astype('uint8'), cmap="gray")
    plt.show()


def normalize(image):
    output = image.copy()
    output = output - output.mean()

    return output


def symmetrize_to_fit(input_image, fit_mod):
    n, m = input_image.shape
    n_out = n + fit_mod - n % fit_mod if n % fit_mod != 0 else n
    m_out = m + fit_mod - m % fit_mod if m % fit_mod != 0 else m
    tmp = np.empty((n_out, m_out), dtype=input_image.dtype)
    tmp[:n, :m] = input_image
    tmp[n:, :m] = input_image[(n - 2):(n - 2) - (n_out - n):-1, :]
    tmp[:512, 512:] = input_image[:, (m - 2):(m - 2) - (m_out - n):-1]
    tmp[512:, 512:] = input_image[(n - 2):(n - 2) - (n_out - n):-1, (m - 2):(m - 2) - (m_out - m):-1]

    return tmp
