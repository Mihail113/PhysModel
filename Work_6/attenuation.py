"""Задача 6: светосбор при экспоненциальном затухании в сферическом детекторе."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Source:
    def __init__(self, Y=10000):
        self.Y = Y

    def generate_photons(self, E):
        if E < 0:
            raise ValueError("Энергия не может быть отрицательной")

        photon_count = int(round(self.Y * E))
        vertical_components = np.random.uniform(-1.0, 1.0, photon_count)
        azimuth_angles = np.random.uniform(0.0, 2.0 * np.pi, photon_count)
        radial_components = np.sqrt(1.0 - vertical_components ** 2)

        return np.column_stack((
            radial_components * np.cos(azimuth_angles),
            radial_components * np.sin(azimuth_angles),
            vertical_components,
        )).astype(np.float32)


class Detector:
    def __init__(self, R=2.0, N=40, r=0.1, a=0.5, attenuation_length=8.0):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
        self.attenuation_length = attenuation_length
        if attenuation_length <= 0:
            raise ValueError("Длина затухания должна быть положительной")
        self.pmt_coordinates = self._generate_uniform_pmts()

    def _generate_uniform_pmts(self):
        indices = np.arange(self.N, dtype=float)
        heights = 1.0 - 2.0 * indices / (self.N - 1)
        circle_radii = np.sqrt(1.0 - heights ** 2)
        angles = np.pi * (np.sqrt(5.0) - 1.0) * indices

        return (self.R * np.column_stack((
            circle_radii * np.cos(angles),
            heights,
            circle_radii * np.sin(angles),
        ))).astype(np.float32)

    def find_pmt_for_points_vectorized(self, hit_points):
        if len(hit_points) == 0:
            return np.empty(0, dtype=np.int32)

        # Для точек на сфере ближайший ФЭУ максимизирует скалярное произведение.
        projections = hit_points @ self.pmt_coordinates.T
        closest_indices = np.argmax(projections, axis=1)
        offsets = hit_points - self.pmt_coordinates[closest_indices]
        distance_squared = np.einsum("ij,ij->i", offsets, offsets)
        return np.where(distance_squared <= self.r ** 2, closest_indices, -1)

    def simulate_event(self, source_pos, directions, use_attenuation=True):
        if len(directions) == 0:
            return 0

        emission_point = np.asarray(source_pos, dtype=np.float32)
        ray_vectors = np.asarray(directions, dtype=np.float32)

        # |emission_point + t * ray_vectors|^2 = R^2.
        # Направления нормированы, поэтому коэффициент при t^2 равен 1.
        linear_coefficients = 2.0 * (ray_vectors @ emission_point)
        constant_coefficient = float(emission_point @ emission_point) - self.R ** 2
        discriminants = linear_coefficients ** 2 - 4.0 * constant_coefficient
        path_lengths = (-linear_coefficients + np.sqrt(np.maximum(discriminants, 0.0))) / 2.0

        # При источнике внутри сферы все лучи пересекают границу вперёд.
        forward_rays = (discriminants >= 0.0) & (path_lengths >= 0.0)
        ray_vectors = ray_vectors[forward_rays]
        path_lengths = path_lengths[forward_rays]
        if path_lengths.size == 0:
            return 0

        if use_attenuation:
            transmission = np.exp(-path_lengths / self.attenuation_length)
            remains = np.random.random(path_lengths.size) < transmission
            ray_vectors = ray_vectors[remains]
            path_lengths = path_lengths[remains]
            if path_lengths.size == 0:
                return 0

        impacts = emission_point + path_lengths[:, None] * ray_vectors
        closest_pmts = self.find_pmt_for_points_vectorized(impacts)
        registered_photons = np.count_nonzero(closest_pmts >= 0)
        if registered_photons == 0:
            return 0

        photoelectron_samples = np.random.poisson(self.a, size=registered_photons)
        return int(photoelectron_samples.sum())


def run_experiment(detector, source, energy, z_positions, events_per_position):
    attenuated_results = {}
    reference_results = {}

    for height in z_positions:
        position = (0.0, 0.0, height)
        charges_with_loss = np.empty(events_per_position, dtype=np.int32)
        charges_without_loss = np.empty(events_per_position, dtype=np.int32)

        for event_index in range(events_per_position):
            # Одни и те же начальные направления для двух вариантов.
            photon_directions = source.generate_photons(energy)
            charges_with_loss[event_index] = detector.simulate_event(
                position, photon_directions, use_attenuation=True
            )
            charges_without_loss[event_index] = detector.simulate_event(
                position, photon_directions, use_attenuation=False
            )

        attenuated_results[height] = charges_with_loss
        reference_results[height] = charges_without_loss

        mean_loss = charges_with_loss.mean()
        mean_reference = charges_without_loss.mean()
        reduction = (1.0 - mean_loss / mean_reference) * 100.0
        print(f"z = {height:>3.1f} м: ")
        print(f"  С затуханием:    {mean_loss:>7.2f} ± {charges_with_loss.std():>6.2f} ф.э.")
        print(f"  Без затухания:   {mean_reference:>7.2f} ± {charges_without_loss.std():>6.2f} ф.э.")
        print(f"  Снижение заряда: {reduction:>7.2f}%\n")

    return attenuated_results, reference_results


def plot_results(attenuated, reference):
    heights = list(attenuated)
    averages_with = [attenuated[z].mean() for z in heights]
    averages_without = [reference[z].mean() for z in heights]

    figure, panels = plt.subplots(1, 3, figsize=(17, 5))

    for panel, height in zip(panels[:2], (heights[0], heights[-1])):
        panel.hist(reference[height], bins=25, alpha=0.6, label="Без затухания")
        panel.hist(attenuated[height], bins=25, alpha=0.6, label="С затуханием")
        panel.set_title(f"Распределение заряда, z = {height:.1f} м")
        panel.set_xlabel("Полный заряд (ф.э.)")
        panel.set_ylabel("Количество событий")
        panel.legend()

    panels[2].plot(heights, averages_without, "o-", label="Без затухания")
    panels[2].plot(heights, averages_with, "s-", label="С затуханием")
    panels[2].set_title("Средний светосбор вдоль оси z")
    panels[2].set_xlabel("Положение источника z (м)")
    panels[2].set_ylabel("Средний заряд (ф.э.)")
    panels[2].legend()

    for panel in panels:
        panel.grid(alpha=0.3, linestyle="--")

    figure.tight_layout()
    image_path = Path(__file__).with_name("attenuation_comparison.png")
    figure.savefig(image_path, dpi=150)
    print(f"Графики сохранены в {image_path}")
    if "agg" in plt.get_backend().lower():
        plt.close(figure)
    else:
        plt.show()


def test_model():
    detector = Detector()
    source = Source()
    assert source.generate_photons(0).shape == (0, 3)
    assert detector.pmt_coordinates.shape == (40, 3)
    assert np.allclose(np.linalg.norm(detector.pmt_coordinates, axis=1), detector.R)

    test_directions = source.generate_photons(0.01)
    assert np.allclose(np.linalg.norm(test_directions, axis=1), 1.0, atol=1e-6)
    assert detector.simulate_event([0, 0, 0], test_directions, False) >= 0
    assert detector.simulate_event([0, 0, 0], test_directions, True) >= 0
    print("Базовые тесты пройдены")


def main():
    np.random.seed(42)
    detector = Detector(R=2.0, N=40, r=0.1, a=0.5, attenuation_length=8.0)
    source = Source(Y=10000)
    energy = 1.0
    event_count = 1000
    source_heights = [0.0, 0.5, 1.0, 1.5, 1.8]

    print(f"Энергия: {energy} МэВ; длина затухания: {detector.attenuation_length} м")
    print(f"Для каждого положения моделируется {event_count} событий.\n")
    charges_with, charges_without = run_experiment(
        detector, source, energy, source_heights, event_count
    )

    center_ratio = charges_with[0.0].mean() / charges_without[0.0].mean()
    expected_center_ratio = np.exp(-detector.R / detector.attenuation_length)
    print("Проверка для источника в центре:")
    print(f"  Теоретическое отношение зарядов: {expected_center_ratio:.4f}")
    print(f"  Отношение в моделировании:       {center_ratio:.4f}")
    print("Изменение среднего при смещении зависит от геометрии и положения ФЭУ.")

    plot_results(charges_with, charges_without)
    test_model()


if __name__ == "__main__":
    main()
