import numpy as np


class SmartSource:
    def __init__(self, Y=10000):
        self.Y = Y

    def generate_photons_generator(self, E, chunk_size=100):
        if E < 0:
            raise ValueError("Энергия E не может быть отрицательной")
        if chunk_size <= 0:
            raise ValueError("Размер порции должен быть положительным")

        average_count = E * self.Y
        number_to_emit = np.random.poisson(average_count)
        remaining = number_to_emit

        while remaining > 0:
            batch_length = min(remaining, chunk_size)

            cos_polar_angle = np.random.uniform(-1.0, 1.0, batch_length)
            azimuth_angle = np.random.uniform(0.0, 2.0 * np.pi, batch_length)
            transverse_component = np.sqrt(1.0 - cos_polar_angle**2)

            photon_vectors = np.column_stack(
                (
                    transverse_component * np.cos(azimuth_angle),
                    transverse_component * np.sin(azimuth_angle),
                    cos_polar_angle,
                )
            ).astype(np.float32)

            remaining -= batch_length
            yield photon_vectors


def test_source():
    np.random.seed(123)
    source = SmartSource(Y=10000)
    samples = list(source.generate_photons_generator(E=0.15, chunk_size=100))

    assert samples
    assert all(batch.shape[1] == 3 for batch in samples)
    assert all(batch.shape[0] == 100 for batch in samples[:-1])
    assert 1 <= samples[-1].shape[0] <= 100
    assert all(batch.dtype == np.float32 for batch in samples)

    vectors = np.concatenate(samples)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    assert list(source.generate_photons_generator(E=0.0)) == []

    print("Все тесты пройдены успешно")


if __name__ == "__main__":
    np.random.seed(42)
    source = SmartSource(Y=10000)
    event_energy = 0.15

    print(f"Энергия события: {event_energy} МэВ")
    print(f"Ожидаемое среднее число фотонов: {event_energy * source.Y:.0f}\n")
    print("--- Генерация фотонов порциями ---")

    counted_photons = 0
    batches_received = 0

    for vectors in source.generate_photons_generator(event_energy, chunk_size=100):
        batches_received += 1
        counted_photons += len(vectors)
        print(
            f"Порция №{batches_received:<2} | Фотонов: {len(vectors):<3} | "
            f"Всего: {counted_photons}"
        )
        if batches_received == 1:
            print(f"Форма массива: {vectors.shape}; первый вектор: {vectors[0]}")

    print("--- Генерация завершена ---")
    print(f"Количество порций: {batches_received}")
    print(f"Общее число фотонов: {counted_photons}\n")

    test_source()
