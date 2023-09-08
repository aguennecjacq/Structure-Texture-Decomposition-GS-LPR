from numba import float64, jit, void, int64
from numba.types import UniTuple
import numpy as np
from proximal_operators import D, svt, prox_tv
from Q_operator import Q, Q_inv
from P_operator import get_weights_map, P, P_inv


@jit(float64(float64[:, :], float64), cache=True, nopython=True)
def grad_sparsity_estimation(input_array, percent):
    norm_grad = np.sqrt(np.sum(D(input_array) ** 2, axis=-1)).ravel()
    norm_grad = np.sort(norm_grad)[::-1]
    cst = norm_grad.sum() * percent
    return np.sum(norm_grad.cumsum() < cst) + 1


@jit(float64(float64[:, :], float64), cache=True, nopython=True)
def rank_estimation(input_array, percent):
    u, d, v = np.linalg.svd(input_array, full_matrices=False)   # we should use the option compute_uv=False, but jit doesn't work with this option
    cst = d.sum() * percent
    return np.sum(d.cumsum() <= cst)


@jit(void(float64[:, :, :], float64[:], float64, float64), fastmath=True, cache=True, nopython=True)
def update_mu(q_array, mu, percent, cst):
    for k in range(mu.shape[0]):
        grad_sparsity = grad_sparsity_estimation(q_array[..., k], percent)
        mu[k] = cst / np.sqrt(grad_sparsity)


@jit(void(float64[:, :, :], float64[:], float64), cache=True, nopython=True)
def update_gamma(p_array, gamma, percent):
    for k in range(gamma.shape[0]):
        rank = rank_estimation(p_array[..., k], percent)
        if rank > 0:
            gamma[k] = 1.0 / np.sqrt(rank)
        else:
            gamma[k] = 10.0


@jit(UniTuple(float64[:, :], 2)(float64[:, :], int64, int64, int64, int64, int64, float64), cache=True, nopython=True)
def std_tv_low_rank(input_image, r=5, tile_size=64, rho=5.0, overlap=16, nb_iter=200, update_cst=0.65):
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
    step_r = (r // 2) + 1
    nb_patches = (tile_shape[1] // step_r) * (tile_shape[1] // step_r)
    weights_map = get_weights_map(tile_shape, r)  # weights map for the reconstruction of the patch_operator

    # memory allocations
    q_array = np.zeros((tile_shape[0], tile_shape[1], grid_size))
    p_array = np.zeros((r ** 2, nb_patches, grid_size))
    mu = np.zeros(grid_size)
    gamma = np.ones(grid_size) / r

    # A good initial guess of mu is to set it using the grad_sparsity of the initial image
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
            update_mu(q_array, mu, percent, update_cst)     # we update mu here to avoid unnecessary computations
        Q_inv(q_array, u, overlap)

        # update v
        tmp = f - u - y
        h_v = gamma / rho
        Q(tmp, q_array, overlap)
        P(q_array, p_array, r)
        svt(p_array, tau=h_v)
        if iter_ % 10 == 0:
            update_gamma(p_array, gamma, percent)           # we update gamma here to avoid unnecessary computations
        q_array = np.zeros((tile_shape[0], tile_shape[1], grid_size))  # we must reset memory of the q_array to avoid memory leakage
        P_inv(p_array, q_array, r, weights_map)
        Q_inv(q_array, v, overlap)

        # update y
        y = y + (u + v - f)

    return u, v
