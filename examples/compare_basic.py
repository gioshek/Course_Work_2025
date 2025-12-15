import matplotlib.pyplot as plt
from pathlib import Path
from src.experiments import save_metrics_csv
from src.experiments import run_all_frequencies
from src.renderer import (
    show_images,
    show_fft,
    export_noise_pdf,
    export_fft_pdf,
)

# =========================
# ПУТИ
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# НАСТРОЙКИ
# =========================


SIZE = 256
SEED = 42

FREQUENCY_PRESETS = {
    "low": {
        "scale": 128.0,
        "octaves": 2,
        "lacunarity": 2.0,
        "gain": 0.5,
    },
    "mid": {
        "scale": 64.0,
        "octaves": 4,
        "lacunarity": 2.0,
        "gain": 0.5,
    },
    "high": {
        "scale": 32.0,
        "octaves": 6,
        "lacunarity": 2.0,
        "gain": 0.5,
    },
}

# =========================
# ВЫВОД
# =========================

def print_table(results: dict) -> None:
    print("\n=== METRICS ===")
    for name, r in results.items():
        print(
            f"{name:12s} | "
            f"Entropy: {r['entropy']:>9.3f} | "
            f"Autocorr: {r['autocorr']:>6.3f} | "
            f"Time: {r['time'] * 1000:>7.1f} ms"
        )

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    all_images, all_results = run_all_frequencies(
        size=SIZE,
        seed=SEED,
        frequency_presets=FREQUENCY_PRESETS,
    )

    # --- интерактивный просмотр ---
    for freq in FREQUENCY_PRESETS:
        show_images(all_images[freq], f"{freq.upper()} FREQUENCY")
        show_fft(all_images[freq], f"{freq.upper()} FREQUENCY")

        print_table(all_results[freq])

        input("\nPress Enter to continue...")
        plt.close("all")

    # --- PDF ---
    export_noise_pdf(
        all_images,
        FIGURES_DIR / "noise_comparison.pdf",
    )

    export_fft_pdf(
        all_images,
        FIGURES_DIR / "fft_comparison.pdf",
    )

    save_metrics_csv(
        all_results,
        PROJECT_ROOT / "report" / "metrics.csv",
    )
