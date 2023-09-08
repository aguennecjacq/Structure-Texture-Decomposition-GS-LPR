from numba import jit, float64, int64, void
from numba.types import UniTuple
import numpy as np


@jit(float64[:, :](UniTuple(int64, 2), int64), cache=True, nopython=True)
def get_weights_map(array_shape=(64, 64), r=5):
    step = (r // 2) + 1
    n, m = array_shape
    grid_x = (n // step)
    grid_y = (m // step)

    weights_map = np.zeros(array_shape, dtype=float64)
    for i in range(grid_x):
        for j in range(grid_y):
            pos_x = i * step
            pos_y = j * step
            if pos_x + r > n:
                pos_x = n - r
            if pos_y + r > m:
                pos_y = m - r
            weights_map[pos_x: pos_x + r, pos_y:pos_y + r] += 1

    weights_map **= (-1)
    return weights_map


@jit(void(float64[:, :, :], float64[:, :, :], int64), cache=True, nopython=True)
def P(input_array, output, p):
    n, m = input_array.shape[:2]
    step = (p // 2) + 1

    grid_x = (n // step)
    grid_y = (m // step)

    # i follow the patch pattern, j indexes the position we are currently at and k the depth.
    for patch_index in range(p ** 2):
        for pos_index in range(grid_x * grid_y):
            for k in range(output.shape[2]):
                # retrieve position on the image from the position index
                x_index = pos_index // grid_y
                y_index = pos_index - x_index * grid_y
                pos_x = x_index * step
                pos_y = y_index * step

                # retrieve the position (i,j) on the current patch
                i = patch_index // p
                j = patch_index - i * p

                if pos_x + p > n:
                    pos_x = n - p
                if pos_y + p > m:
                    pos_y = m - p
                output[patch_index, pos_index, k] = input_array[pos_x + i, pos_y + j, k]


@jit(void(float64[:, :, :], float64[:, :, :], int64, float64[:, :]), cache=True, nopython=True)
def P_inv(input_array, output, r, weights_map):
    n, m = output.shape[:2]
    step = (r // 2) + 1
    grid_x = (n // step)
    grid_y = (m // step)

    for patch_index in range(r ** 2):
        for pos_index in range(grid_x * grid_y):
            for k in range(output.shape[2]):
                # retrieve position on the image from the position index
                x_index = pos_index // grid_y
                y_index = pos_index - x_index * grid_y
                pos_x = x_index * step
                pos_y = y_index * step

                # retrieve the position (i,j) on the current patch
                i = patch_index // r
                j = patch_index - i * r

                if pos_x + r > n:
                    pos_x = n - r
                if pos_y + r > m:
                    pos_y = m - r
                output[pos_x + i, pos_y + j, k] += input_array[patch_index, pos_index, k] * weights_map[pos_x + i, pos_y + j]