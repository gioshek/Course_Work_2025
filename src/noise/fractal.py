import numpy as np
from typing import Callable


def fbm(
    noise_fn: Callable[..., np.ndarray],
    width: int,
    height: int,
    *,
    scale: float = 64.0,
    octaves: int = 4,
    lacunarity: float = 2.0,
    gain: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """
    Fractal Brownian Motion (fBm)

    noise_fn: базовая функция шума (perlin_noise, value_noise, simplex_noise)
    """

    result = np.zeros((height, width), dtype=np.float64)

    amplitude = 1.0
    frequency = 1.0
    max_amplitude = 0.0

    for i in range(octaves):
        noise = noise_fn(
            width,
            height,
            scale=scale / frequency,
            seed=seed + i * 17,
        )

        result += noise * amplitude
        max_amplitude += amplitude

        amplitude *= gain
        frequency *= lacunarity

    # Нормализация
    result /= max_amplitude

    return result
