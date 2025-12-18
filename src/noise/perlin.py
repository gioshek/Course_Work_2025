import numpy as np


# =========================
# Вспомогательные функции
# =========================

def fade(t):
    """Сглаживающая функция Перлина"""
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def lerp(a, b, t):
    """Линейная интерполяция"""
    return a + t * (b - a)


def gradient(hash, x, y):
    """Градиентное скалярное произведение"""
    vectors = np.array([
        [1, 1], [-1, 1], [1, -1], [-1, -1],
        [1, 0], [-1, 0], [0, 1], [0, -1],
    ])
    g = vectors[hash % 8]
    return g[..., 0] * x + g[..., 1] * y


# =========================
# Perlin noise 2D
# =========================

def perlin_noise(
    width: int,
    height: int,
    scale: float = 10.0,
    seed: int = 0
) -> np.ndarray:
    """
    Генерация 2D Perlin noise

    :param width: ширина изображения
    :param height: высота изображения
    :param scale: частота шума (меньше — крупнее структуры)
    :param seed: зерно генератора
    :return: numpy array [0, 1]
    """

    np.random.seed(seed)

    # координатная сетка
    x = np.linspace(0, scale, width, endpoint=False)
    y = np.linspace(0, scale, height, endpoint=False)
    x, y = np.meshgrid(x, y)

    xi = x.astype(int)
    yi = y.astype(int)

    xf = x - xi
    yf = y - yi

    # таблица перестановок
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.concatenate([p, p])

    # хэши углов ячейки
    aa = p[p[xi % 256] + yi % 256]
    ab = p[p[xi % 256] + (yi + 1) % 256]
    ba = p[p[(xi + 1) % 256] + yi % 256]
    bb = p[p[(xi + 1) % 256] + (yi + 1) % 256]

    # градиенты + интерполяция
    x1 = gradient(aa, xf, yf)
    x2 = gradient(ba, xf - 1, yf)
    u = fade(xf)
    y1 = lerp(x1, x2, u)

    x1 = gradient(ab, xf, yf - 1)
    x2 = gradient(bb, xf - 1, yf - 1)
    y2 = lerp(x1, x2, u)

    v = fade(yf)
    noise = lerp(y1, y2, v)

    # нормализация в [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise
