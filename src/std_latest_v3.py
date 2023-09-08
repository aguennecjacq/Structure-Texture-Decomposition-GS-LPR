from numba import float64, int64, jit, void
from numba.types import UniTuple
import numpy as np
from math import ceil


@jit(float64[:, :](UniTuple(int64, 2), int64), cache=True, nopython=True)
def get_weights_map(array_shape=(50, 50), r=5):
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


@jit(void(float64[:, :, :], UniTuple(int64, 2), int64), cache=True, nopython=True)
def interpolate(input_array, grid=(10, 10), s=10):
    n, m = input_array.shape[:2]

    for i in range(grid[0]):
        for j in range(grid[1]):

            k = i * grid[1] + j

            if (i, j) == (0, 0):
                orientation = (False, True, False, True)
            elif (i, j) == (grid[0] - 1, 0):
                orientation = (True, False, False, True)
            elif (i, j) == (0, grid[1] - 1):
                orientation = (False, True, True, False)
            elif (i, j) == (grid[0] - 1, grid[1] - 1):
                orientation = (True, False, True, False)

            # Sides
            elif i == 0:
                orientation = (False, True, True, True)
            elif i == (grid[0] - 1):
                orientation = (True, False, True, True)
            elif j == 0:
                orientation = (True, True, False, True)
            elif j == (grid[1] - 1):
                orientation = (True, True, True, False)

            # General case
            else:
                orientation = (True, True, True, True)
            for z in range(s):
                # TOP
                if orientation[0]:
                    for y in range(m):
                        input_array[z, y, k] *= (z + 1.0) / (s + 1.0)

                # BOTTOM
                if orientation[1]:
                    for y in range(m):
                        input_array[n - z - 1, y, k] *= (z + 1.0) / (s + 1.0)

                # LEFT
                if orientation[2]:
                    for x in range(n):
                        input_array[x, z, k] *= (z + 1.0) / (s + 1.0)

                # RIGHT
                if orientation[3]:
                    for x in range(n):
                        input_array[x, m - z - 1, k] *= (z + 1.0) / (s + 1.0)


@jit(void(float64[:, :], float64[:, :, :], int64), cache=True, nopython=True)
def Q(input_array, output, sep=10):
    n, m = input_array.shape
    t1, t2 = output.shape[:2]

    step_x = t1 - sep
    step_y = t2 - sep
    grid_x = (n // step_x) + 1
    grid_y = (m // step_y) + 1

    for i in range(grid_x):
        for j in range(grid_y):
            for z in range(t1 * t2):
                pos_x = i * step_x
                pos_y = j * step_y
                k = i * grid_y + j
                x = z // t2
                y = z - x * t2
                if pos_x + x < n and pos_y + y < m:
                    output[x, y, k] = input_array[pos_x + x, pos_y + y]
                else:
                    output[x, y, k] = 0


@jit(void(float64[:, :, :], float64[:, :], int64), cache=True, nopython=True)
def Q_inv(input_array, output, sep=10):
    n, m = output.shape
    step_x = input_array.shape[0] - sep
    step_y = input_array.shape[1] - sep
    grid_y = (m // step_y) + 1

    for x in range(n):
        for y in range(m):
            i = x // step_x
            j = y // step_y
            # edge case

            k = i * grid_y + j

            # works when we are without a border at the end of the image
            if (x % step_x >= sep) and (y % step_y >= sep):
                output[x, y] = input_array[x % step_x, y % step_y, k]
            elif (x < step_x) and (y < step_y):
                output[x, y] = input_array[x % step_x, y % step_y, k]
            elif (x < step_x) and (y % step_y >= sep):
                output[x, y] = input_array[x % step_x, y % step_y, k]
            elif (x % step_x >= sep) and (y < step_y):
                output[x, y] = input_array[x % step_x, y % step_y, k]
            elif ((x % step_x >= sep) or (x < step_x)) and y % step_y < sep:
                output[x, y] = input_array[x % step_x, step_y + y % step_y, k - 1] + input_array[
                    x % step_x, y % step_y, k]
            elif x % step_x < sep and ((y % step_y >= sep) or (y < step_y)):
                output[x, y] = input_array[x % step_x, y % step_y, k] + input_array[
                    step_x + x % step_x, y % step_y, k - grid_y]

            elif (x % step_x < sep) and (y % step_y < sep):
                output[x, y] = input_array[step_x + x % step_x, y % step_y, k - grid_y] + input_array[
                    step_x + x % step_x, step_y + y % step_y, k - grid_y - 1] + input_array[
                                   x % step_x, step_y + y % step_y, k - 1] + input_array[x % step_x, y % step_y, k]


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


@jit(void(float64[:, :, :], float64[:]), cache=True, nopython=True)
def svt(input_array, tau):
    for k in range(input_array.shape[-1]):
        u, d, v = np.linalg.svd(input_array[..., k], full_matrices=False)
        d_ = np.zeros_like(d)
        for idx in range(len(d)):
            if d[idx] > tau[k]:
                d_[idx] = d[idx] - tau[k]
            else:
                break
        input_array[..., k] = u @ np.diag(d_) @ v


@jit(float64[:, :, :](float64[:, :]), fastmath=True, nopython=True)
def D(input_array):
    output = np.ascontiguousarray(np.zeros(input_array.shape + (2,)))
    output[:-1, :, 0] = input_array[1:, :] - input_array[:-1, :]  # grad x
    output[:, :-1, 1] = input_array[:, 1:] - input_array[:, :-1]  # grad y
    return output


@jit(float64[:, :](float64[:, :, :]), fastmath=True, nopython=True)
def div(input_array):
    div_x = np.ascontiguousarray(np.zeros(input_array.shape[:2]))
    div_y = np.ascontiguousarray(np.zeros(input_array.shape[:2]))

    div_x[1:-1, :] = input_array[1:-1, :, 0] - input_array[:-2, :, 0]
    div_x[0, :] = input_array[0, :, 0]
    div_x[-1, :] = -input_array[-2, :, 0]

    div_y[:, 1:-1] = input_array[:, 1:-1, 1] - input_array[:, :-2, 1]
    div_y[:, 0] = input_array[:, 0, 1]
    div_y[:, -1] = -input_array[:, -2, 1]

    return div_x + div_y


@jit(float64[:, :, :](float64[:, :, :], float64[:, :], float64), fastmath=True, cache=True,
     nopython=True)
def grad_F_tv(y, x, h):
    return 2 * D(x / h - div(y))


@jit(float64[:, :, :](float64[:, :, :]), fastmath=True, cache=True, nopython=True)
def project_unit(z):
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            norm_z = np.sqrt(z[i, j, 0] ** 2 + z[i, j, 1] ** 2)
            if norm_z > 1:
                z[i, j, 0] = z[i, j, 0] / norm_z
                z[i, j, 1] = z[i, j, 1] / norm_z
    return z


@jit(float64[:, :](float64[:, :], float64, int64), fastmath=True, cache=True, nopython=True)
def _prox_tv(x, h, nb_iter=200):
    w = np.ascontiguousarray(np.zeros(x.shape + (2,)))
    y = w
    t = 1
    h_tv = 0.08  # optimal step for the computations
    # Accelerated Projection algorithm
    for _ in range(nb_iter):
        w_prev = w
        w = project_unit(y - h_tv * grad_F_tv(y, x, h))
        t_prev = t
        t = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y = w + ((t_prev - 1) / t) * (w - w_prev)
    return x - h * div(w)


@jit(void(float64[:, :, :], float64[:], int64), nopython=True)
def prox_tv(x, h, nb_iter=200):
    for idk in range(x.shape[-1]):
        x[..., idk] = _prox_tv(x[..., idk], h[idk], nb_iter=nb_iter)


@jit(float64(float64[:, :], float64), cache=True, nopython=True)
def sparse_estimation(input_array, percent):
    norm_grad = np.sqrt(np.sum(D(input_array) ** 2, axis=-1)).ravel()
    norm_grad = np.sort(norm_grad)[::-1]
    cst = norm_grad.sum() * percent

    return np.sum(norm_grad.cumsum() < cst) + 1


@jit(float64(float64[:, :], float64), cache=True, nopython=True)
def rank_estimation(input_array, percent):
    u, d, v = np.linalg.svd(input_array, full_matrices=False)
    cst = d.sum() * percent
    return np.sum(d.cumsum() < cst) + 1


@jit(void(float64[:, :, :], float64[:], float64, float64), fastmath=True, cache=True, nopython=True)
def update_mu(q_array, mu, percent, cst):
    for k in range(mu.shape[0]):
        sparsity = sparse_estimation(q_array[..., k], percent)
        mu[k] = cst / np.sqrt(sparsity)


@jit(void(float64[:, :, :], float64[:], float64), cache=True, nopython=True)
def update_gamma(p_array, gamma, percent):
    for k in range(gamma.shape[0]):
        rank = rank_estimation(p_array[..., k], percent)
        if rank > 0:
            gamma[k] = 1.0 / np.sqrt(rank)
        else:
            gamma[k] = 1.0


@jit(UniTuple(float64[:, :], 2)(float64[:, :], int64, int64, int64, int64, int64, float64), cache=True, nopython=True)
def std_tv_low_rank(input_image, p=5, tile_size=64, rho=5.0, overlap=20, nb_iter=40, update_cst=0.6):
    n, m = input_image.shape
    percent = 0.9
    tile_shape = (tile_size, tile_size)

    # init
    f = input_image
    u = np.zeros_like(f)
    v = np.zeros_like(f)
    y = np.zeros_like(f)

    # setup parameters
    step_x = tile_shape[0] - overlap
    step_y = tile_shape[1] - overlap
    grid = ((n // step_x) + 1, (m // step_y) + 1)
    grid_size = grid[0] * grid[1]
    step_r = (p // 2) + 1
    nb_patches = (tile_shape[1] // step_r) * (tile_shape[1] // step_r)
    weights_map = get_weights_map(tile_shape, p)  # weights map for the reconstruction of p_array -> q_array

    # memory allocations
    q_array = np.zeros((tile_shape[0], tile_shape[1], grid_size))
    p_array = np.zeros((p ** 2, nb_patches, grid_size))

    mu = np.zeros(grid_size)
    gamma = np.ones(grid_size) / p

    Q(f, q_array, overlap)
    update_mu(q_array, mu, percent, update_cst)
    q_array = np.zeros((tile_shape[0], tile_shape[1], grid_size))

    for iter_ in range(nb_iter):

        # update u
        tmp = f - v - y
        h_u = mu / rho
        Q(tmp, q_array, overlap)
        prox_tv(q_array, h=h_u, nb_iter=200)
        if iter_ % 10 == 0:
            update_mu(q_array, mu, percent, update_cst)  # we update mu here to avoid unnecessary computations
        interpolate(q_array, grid, overlap)
        Q_inv(q_array, u, overlap)

        # update v
        tmp = f - u - y
        h_v = gamma / rho
        Q(tmp, q_array, overlap)
        P(q_array, p_array, p)
        svt(p_array, tau=h_v)  # unfortunately, cpu version is faster than gpu version
        if iter_ % 10 == 0:
            update_gamma(p_array, gamma, percent)  # we update gamma here to avoid unnecessary computations
        q_array = np.zeros((tile_shape[0], tile_shape[1], grid_size))  # we must reset memory of the q_array  for some reason to avoid memory leakage
        P_inv(p_array, q_array, p, weights_map)

        interpolate(q_array, grid, overlap)
        Q_inv(q_array, v, overlap)

        # update y
        y = y + (u + v - f)

    return u, v


if __name__ == "__main__":
    from misc import save_img, get_img
    import os
    import sys

    image_name = sys.argv[1]
    print(image_name)
    image_path = "../images/" + image_name
    input_image_file = get_img(image_path).astype(np.float64)
    tata = std_tv_low_rank(input_image_file, p=5, tile_size=64, rho=5.0, overlap=16, nb_iter=200, update_cst=0.65)

    output_folder = image_name[:-4]
    try:
        os.mkdir(output_folder)
    except FileExistsError:
        pass

    save_img(tata[0], f"./{output_folder}/cartoon.png")
    save_img(tata[1] + 0.5, f"./{output_folder}/texture.png")
