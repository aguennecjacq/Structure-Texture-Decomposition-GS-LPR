from numba import jit, float64, void, int64
import numpy as np


@jit(void(float64[:, :, :], float64[:]), cache=True, nopython=True)
def svt(input_array, tau):
    for k in range(input_array.shape[-1]):
        u, d, v = np.linalg.svd(input_array[..., k],
                                full_matrices=False)  # ideally we should use the option compute_uv=False, but jit doesn't allow us this
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
