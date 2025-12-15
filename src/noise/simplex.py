import numpy as np
from math import floor


# Градиенты для 2D Simplex
_GRADIENTS = np.array([
    [1, 1], [-1, 1], [1, -1], [-1, -1],
    [1, 0], [-1, 0], [0, 1], [0, -1]
], dtype=np.float64)


def _hash(ix: int, iy: int, seed: int) -> int:
    h = ix * 374761393 + iy * 668265263 + seed * 1442695040888963407
    h = (h ^ (h >> 13)) * 1274126177
    return h & 7  # 0..7


def _dot(g, x, y):
    return g[0] * x + g[1] * y


def simplex_noise(width: int, height: int, scale: float = 64.0, seed: int = 0) -> np.ndarray:
    """
    2D Simplex noise
    """
    F2 = 0.5 * (np.sqrt(3.0) - 1.0)
    G2 = (3.0 - np.sqrt(3.0)) / 6.0

    noise = np.zeros((height, width), dtype=np.float64)

    for y in range(height):
        for x in range(width):
            # Масштабируем координаты
            fx = x / scale
            fy = y / scale

            # Skew
            s = (fx + fy) * F2
            i = floor(fx + s)
            j = floor(fy + s)

            t = (i + j) * G2
            X0 = i - t
            Y0 = j - t

            x0 = fx - X0
            y0 = fy - Y0

            # Определяем второй угол симплекса
            if x0 > y0:
                i1, j1 = 1, 0
            else:
                i1, j1 = 0, 1

            x1 = x0 - i1 + G2
            y1 = y0 - j1 + G2
            x2 = x0 - 1.0 + 2.0 * G2
            y2 = y0 - 1.0 + 2.0 * G2

            # Градиенты
            gi0 = _hash(i, j, seed)
            gi1 = _hash(i + i1, j + j1, seed)
            gi2 = _hash(i + 1, j + 1, seed)

            n0 = n1 = n2 = 0.0

            t0 = 0.5 - x0 * x0 - y0 * y0
            if t0 > 0:
                t0 *= t0
                n0 = t0 * t0 * _dot(_GRADIENTS[gi0], x0, y0)

            t1 = 0.5 - x1 * x1 - y1 * y1
            if t1 > 0:
                t1 *= t1
                n1 = t1 * t1 * _dot(_GRADIENTS[gi1], x1, y1)

            t2 = 0.5 - x2 * x2 - y2 * y2
            if t2 > 0:
                t2 *= t2
                n2 = t2 * t2 * _dot(_GRADIENTS[gi2], x2, y2)

            noise[y, x] = 70.0 * (n0 + n1 + n2)

    return noise
