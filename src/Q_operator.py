from numba import void, jit, float64, int64
from numba.types import UniTuple


@jit(void(float64[:, :, :], UniTuple(int64, 2), int64), cache=True, nopython=True)
def interpolate(input_array, grid=(64, 64), overlap=16):
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
            for z in range(overlap):
                # TOP
                if orientation[0]:
                    for y in range(m):
                        input_array[z, y, k] *= (z + 1.0) / (overlap + 1.0)

                # BOTTOM
                if orientation[1]:
                    for y in range(m):
                        input_array[n - z - 1, y, k] *= (z + 1.0) / (overlap + 1.0)

                # LEFT
                if orientation[2]:
                    for x in range(n):
                        input_array[x, z, k] *= (z + 1.0) / (overlap + 1.0)

                # RIGHT
                if orientation[3]:
                    for x in range(n):
                        input_array[x, m - z - 1, k] *= (z + 1.0) / (overlap + 1.0)


@jit(void(float64[:, :], float64[:, :, :], int64), cache=True, nopython=True)
def Q(input_array, output, overlap=16):
    n, m = input_array.shape
    t1, t2 = output.shape[:2]

    step_x = t1 - overlap
    step_y = t2 - overlap
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
def Q_inv(input_array, output, overlap=16):
    n, m = output.shape
    step_x = input_array.shape[0] - overlap
    step_y = input_array.shape[1] - overlap
    grid_y = (m // step_y) + 1
    grid = ((n // step_x) + 1, (m // step_y) + 1)
    interpolate(input_array, grid=grid, overlap=overlap)

    for x in range(n):
        for y in range(m):
            i = x // step_x
            j = y // step_y
            k = i * grid_y + j
            if (x % step_x >= overlap) and (y % step_y >= overlap):
                output[x, y] = input_array[x % step_x, y % step_y, k]
            elif (x < step_x) and (y < step_y):
                output[x, y] = input_array[x % step_x, y % step_y, k]
            elif (x < step_x) and (y % step_y >= overlap):
                output[x, y] = input_array[x % step_x, y % step_y, k]
            elif (x % step_x >= overlap) and (y < step_y):
                output[x, y] = input_array[x % step_x, y % step_y, k]
            elif ((x % step_x >= overlap) or (x < step_x)) and y % step_y < overlap:
                output[x, y] = input_array[x % step_x, step_y + y % step_y, k - 1] + input_array[
                    x % step_x, y % step_y, k]
            elif x % step_x < overlap and ((y % step_y >= overlap) or (y < step_y)):
                output[x, y] = input_array[x % step_x, y % step_y, k] + input_array[
                    step_x + x % step_x, y % step_y, k - grid_y]

            elif (x % step_x < overlap) and (y % step_y < overlap):
                output[x, y] = input_array[step_x + x % step_x, y % step_y, k - grid_y] \
                               + input_array[step_x + x % step_x, step_y + y % step_y, k - grid_y - 1] \
                               + input_array[x % step_x, step_y + y % step_y, k - 1] \
                               + input_array[x % step_x, y % step_y, k]
