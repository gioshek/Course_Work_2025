import numpy as np


def lerp(a, b, t):
    return a + t * (b - a)


def fade(t):
    """Та же fade-функция, что и у Perlin (важно для честного сравнения)"""
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def value_noise(
    width: int,
    height: int,
    scale: float = 10.0,
    seed: int = 0
) -> np.ndarray:
    """
    2D Value Noise

    :param width: ширина изображения
    :param height: высота изображения
    :param scale: частота шума
    :param seed: зерно
    :return: numpy array [0, 1]
    """

    np.random.seed(seed)

    # координаты
    x = np.linspace(0, scale, width, endpoint=False)
    y = np.linspace(0, scale, height, endpoint=False)
    x, y = np.meshgrid(x, y)

    xi = x.astype(int)
    yi = y.astype(int)

    xf = x - xi
    yf = y - yi

    # случайные значения в узлах сетки
    grid = np.random.rand(int(scale) + 2, int(scale) + 2)

    # значения в углах
    v00 = grid[xi, yi]
    v10 = grid[xi + 1, yi]
    v01 = grid[xi, yi + 1]
    v11 = grid[xi + 1, yi + 1]

    # интерполяция
    u = fade(xf)
    v = fade(yf)

    x1 = lerp(v00, v10, u)
    x2 = lerp(v01, v11, u)
    noise = lerp(x1, x2, v)

    # нормализация
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise
