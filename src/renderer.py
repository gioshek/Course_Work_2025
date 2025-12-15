import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from typing import Union

from src.metrics import fft_spectrum


PathLike = Union[str, Path]


# =========================
# ОТОБРАЖЕНИЕ ШУМОВ
# =========================

def show_images(images: dict, title: str) -> None:
    methods = list(images.keys())

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(title, fontsize=16, weight="bold")

    for i, name in enumerate(methods):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(images[name], cmap="gray")
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show(block=False)


# =========================
# ОТОБРАЖЕНИЕ FFT
# =========================

def show_fft(images: dict, title: str) -> None:
    methods = list(images.keys())

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(title + " — FFT", fontsize=16, weight="bold")

    for i, name in enumerate(methods):
        spectrum = fft_spectrum(images[name])
        ax = fig.add_subplot(2, 3, i + 1)
        im = ax.imshow(spectrum, cmap="inferno")
        ax.set_title(name)
        ax.axis("off")

        if i % 3 == 0:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show(block=False)


# =========================
# PDF: ШУМЫ
# =========================

def export_noise_pdf(all_images: dict, path: PathLike) -> None:
    methods = list(next(iter(all_images.values())).keys())

    with PdfPages(str(path)) as pdf:
        for freq, images in all_images.items():
            fig = plt.figure(figsize=(12, 6))
            fig.suptitle(f"{freq.upper()} FREQUENCY", fontsize=16, weight="bold")

            for i, name in enumerate(methods):
                ax = fig.add_subplot(2, 3, i + 1)
                ax.imshow(images[name], cmap="gray")
                ax.set_title(name)
                ax.axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[OK] Noise PDF saved to {path}")


# =========================
# PDF: FFT
# =========================

def export_fft_pdf(all_images: dict, path: PathLike) -> None:
    methods = list(next(iter(all_images.values())).keys())

    with PdfPages(str(path)) as pdf:
        for freq, images in all_images.items():
            fig = plt.figure(figsize=(12, 6))
            fig.suptitle(
                f"{freq.upper()} FREQUENCY — FFT",
                fontsize=16,
                weight="bold",
            )

            for i, name in enumerate(methods):
                spectrum = fft_spectrum(images[name])
                ax = fig.add_subplot(2, 3, i + 1)
                im = ax.imshow(spectrum, cmap="inferno")
                ax.set_title(name)
                ax.axis("off")

                if i % 3 == 0:
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[OK] FFT PDF saved to {path}")
