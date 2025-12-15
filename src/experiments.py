from typing import Dict, Tuple
import numpy as np

from src.noise import (
    perlin_noise,
    value_noise,
    simplex_noise,
    fbm,
)

from src.metrics import (
    entropy,
    autocorrelation,
    benchmark,
)

# =========================
# НАБОР ШУМОВ
# =========================

BASE_NOISES = {
    "Perlin": perlin_noise,
    "Value": value_noise,
    "Simplex": simplex_noise,
}

FBM_NOISES = {
    "fBm-Perlin": perlin_noise,
    "fBm-Value": value_noise,
    "fBm-Simplex": simplex_noise,
}


# =========================
# ОДИН ЭКСПЕРИМЕНТ
# =========================

def run_experiment(
    size: int,
    seed: int,
    base_params: Dict,
    fbm_params: Dict,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
    """
    Запуск одного эксперимента (одна частота)
    """

    images = {}
    results = {}

    # --- базовые шумы ---
    for name, fn in BASE_NOISES.items():
        img = fn(
            size,
            size,
            scale=base_params["scale"],
            seed=seed,
        )

        images[name] = img
        results[name] = {
            "entropy": entropy(img),
            "autocorr": autocorrelation(img),
            "time": benchmark(
                fn,
                size,
                size,
                scale=base_params["scale"],
                seed=seed,
            ),
        }

    # --- fBm ---
    for name, fn in FBM_NOISES.items():
        img = fbm(
            fn,
            size,
            size,
            seed=seed,
            **fbm_params,
        )

        images[name] = img
        results[name] = {
            "entropy": entropy(img),
            "autocorr": autocorrelation(img),
            "time": benchmark(
                fbm,
                fn,
                size,
                size,
                seed=seed,
                **fbm_params,
            ),
        }

    return images, results


# =========================
# ПРОГОН ПО ЧАСТОТАМ
# =========================

def run_all_frequencies(
    size: int,
    seed: int,
    frequency_presets: Dict,
) -> Tuple[Dict, Dict]:
    """
    Запуск эксперимента для LOW / MID / HIGH
    """

    all_images = {}
    all_results = {}

    for freq_name, params in frequency_presets.items():
        print(f"\n=== FREQUENCY: {freq_name.upper()} ===")

        base_params = {
            "scale": params["scale"],
        }

        fbm_params = {
            "scale": params["scale"],
            "octaves": params["octaves"],
            "lacunarity": params["lacunarity"],
            "gain": params["gain"],
        }

        images, results = run_experiment(
            size=size,
            seed=seed,
            base_params=base_params,
            fbm_params=fbm_params,
        )

        all_images[freq_name] = images
        all_results[freq_name] = results

    return all_images, all_results

import csv
from pathlib import Path

def save_metrics_csv(
    all_results: dict,
    output_path: Path,
) -> None:
    """
    Сохраняет метрики в CSV-файл.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Заголовок
        writer.writerow([
            "frequency",
            "method",
            "entropy",
            "autocorr",
            "time_ms",
        ])

        for freq, methods in all_results.items():
            for method, metrics in methods.items():
                writer.writerow([
                    freq,
                    method,
                    round(metrics["entropy"], 6),
                    round(metrics["autocorr"], 6),
                    round(metrics["time"] * 1000, 3),
                ])

