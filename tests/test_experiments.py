from src.experiments import run_all_frequencies


def test_run_all_frequencies_smoke():
    """
    Smoke-тест: проверяем, что эксперимент вообще запускается
    и возвращает данные правильной структуры.
    """

    presets = {
        "low": {
            "scale": 32.0,
            "octaves": 2,
            "lacunarity": 2.0,
            "gain": 0.5,
        }
    }

    images, results = run_all_frequencies(
        size=32,
        seed=0,
        frequency_presets=presets,
    )

    # Проверяем частоту
    assert "low" in images
    assert "low" in results

    # Проверяем шумы
    assert "Perlin" in images["low"]
    assert images["low"]["Perlin"].shape == (32, 32)

    # Проверяем метрики
    metrics = results["low"]["Perlin"]
    assert "entropy" in metrics
    assert "autocorr" in metrics
    assert "time" in metrics
