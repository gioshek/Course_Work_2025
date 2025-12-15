import time
import numpy as np


# =========================
# ЭНТРОПИЯ
# =========================

def entropy(image: np.ndarray) -> float:
    hist, _ = np.histogram(image, bins=256, range=(0, 1), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


# =========================
# FFT СПЕКТР (2D КАРТА)
# =========================

def fft_spectrum(image: np.ndarray) -> np.ndarray:
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    spectrum = np.log1p(magnitude)
    return spectrum


# =========================
# АВТОКОРРЕЛЯЦИЯ (СКАЛЯР)
# =========================

def autocorrelation(image: np.ndarray) -> float:
    mean = np.mean(image)
    var = np.var(image)

    if var == 0:
        return 0.0

    shifted = np.roll(image, shift=1, axis=0)
    value = np.mean((image - mean) * (shifted - mean)) / var
    return float(value)


# =========================
# ПРОИЗВОДИТЕЛЬНОСТЬ
# =========================

def benchmark(func, *args, repeats: int = 3, **kwargs) -> float:
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    return float(np.mean(times))

