"""Задача 7: сравнение циклического и векторизованного моделирования фотонов."""

import argparse
import time

import numpy as np


class SlowDetectorAndSource:
    """Контрольная реализация с обработкой каждого фотона отдельным циклом."""

    def __init__(self, R=2.0, N=40, r=0.1, a=0.5, att_len=8.0,
                 Y=10000, seed=None):
        self.R, self.N, self.r = float(R), int(N), float(r)
        self.a, self.att_len, self.Y = float(a), float(att_len), int(Y)
        self.random = np.random.default_rng(seed)
        self.pmts = []

        spiral_step = np.pi * (np.sqrt(5.0) - 1.0)
        for detector_index in range(self.N):
            height = 1.0 - 2.0 * detector_index / (self.N - 1)
            ring_radius = np.sqrt(1.0 - height ** 2)
            rotation = spiral_step * detector_index
            self.pmts.append([
                self.R * ring_radius * np.cos(rotation),
                self.R * height,
                self.R * ring_radius * np.sin(rotation),
            ])

    def simulate_event_slow(self, E, source_pos):
        photon_count = int(round(self.Y * E))
        collected_charge = 0
        start_x, start_y, start_z = source_pos
        sphere_constant = start_x**2 + start_y**2 + start_z**2 - self.R**2

        for _ in range(photon_count):
            vertical = self.random.uniform(-1.0, 1.0)
            azimuth = self.random.uniform(0.0, 2.0 * np.pi)
            horizontal = np.sqrt(1.0 - vertical**2)
            vx = horizontal * np.cos(azimuth)
            vy = horizontal * np.sin(azimuth)
            vz = vertical

            linear_term = 2.0 * (vx * start_x + vy * start_y + vz * start_z)
            discriminant = linear_term**2 - 4.0 * sphere_constant
            if discriminant < 0.0:
                continue
            distance = (-linear_term + np.sqrt(discriminant)) / 2.0
            if distance < 0.0:
                continue

            transmission = np.exp(-distance / self.att_len)
            if self.random.random() >= transmission:
                continue

            impact_x = start_x + vx * distance
            impact_y = start_y + vy * distance
            impact_z = start_z + vz * distance

            best_distance_sq = float('inf')
            for sensor in self.pmts:
                separation_sq = ((impact_x - sensor[0])**2
                                 + (impact_y - sensor[1])**2
                                 + (impact_z - sensor[2])**2)
                if separation_sq < best_distance_sq:
                    best_distance_sq = separation_sq

            if best_distance_sq <= self.r**2:
                collected_charge += self.random.poisson(self.a)

        return int(collected_charge)


class FastDetectorAndSource:
    """NumPy-реализация: один ndarray направлений на событие."""

    def __init__(self, R=2.0, N=40, r=0.1, a=0.5, att_len=8.0,
                 Y=10000, seed=None):
        self.R, self.N, self.r = float(R), int(N), float(r)
        self.a, self.att_len, self.Y = float(a), float(att_len), int(Y)
        self.random = np.random.default_rng(seed)

        numbers = np.arange(self.N, dtype=np.float64)
        heights = 1.0 - 2.0 * numbers / (self.N - 1)
        ring_sizes = np.sqrt(1.0 - heights**2)
        phases = numbers * np.pi * (np.sqrt(5.0) - 1.0)
        self.pmt_coordinates = (self.R * np.column_stack((
            ring_sizes * np.cos(phases),
            heights,
            ring_sizes * np.sin(phases),
        ))).astype(np.float32)

    def simulate_event_fast(self, E, source_pos):
        photon_count = int(round(self.Y * E))
        if photon_count <= 0:
            return 0

        origin = np.asarray(source_pos, dtype=np.float32)
        polar_cosines = self.random.uniform(-1.0, 1.0, photon_count)
        azimuths = self.random.uniform(0.0, 2.0 * np.pi, photon_count)
        transverse = np.sqrt(1.0 - polar_cosines**2)
        ray_directions = np.column_stack((
            transverse * np.cos(azimuths),
            transverse * np.sin(azimuths),
            polar_cosines,
        )).astype(np.float32)

        linear_terms = 2.0 * (ray_directions @ origin)
        constant_term = float(origin @ origin) - self.R**2
        discriminants = linear_terms**2 - 4.0 * constant_term
        path_lengths = (-linear_terms + np.sqrt(np.maximum(discriminants, 0.0))) / 2.0

        forward = (discriminants >= 0.0) & (path_lengths >= 0.0)
        # Вероятность выживания — exp(-L / attenuation_length).
        survived = forward & (self.random.random(photon_count)
                              < np.exp(-path_lengths / self.att_len))
        if not np.any(survived):
            return 0

        arrival_points = origin + ray_directions[survived] * path_lengths[survived, None]

        # Все центры ФЭУ лежат на сфере одного радиуса.
        # Максимальное скалярное произведение соответствует ближайшему ФЭУ.
        sensor_scores = arrival_points @ self.pmt_coordinates.T
        nearest_sensors = np.argmax(sensor_scores, axis=1)
        differences = arrival_points - self.pmt_coordinates[nearest_sensors]
        squared_offsets = np.einsum('ij,ij->i', differences, differences)
        accepted_count = int(np.count_nonzero(squared_offsets <= self.r**2))
        if accepted_count == 0:
            return 0

        # Независимая пуассоновская генерация для всех попавших фотонов.
        photoelectron_counts = self.random.poisson(self.a, size=accepted_count)
        return int(photoelectron_counts.sum())


def check_implementation():
    reference = SlowDetectorAndSource(seed=11)
    vectorized = FastDetectorAndSource(seed=11)
    assert len(reference.pmts) == vectorized.pmt_coordinates.shape[0] == 40
    assert np.allclose(reference.pmts, vectorized.pmt_coordinates, atol=1e-6)
    assert np.allclose(np.linalg.norm(vectorized.pmt_coordinates, axis=1), 2.0)
    assert reference.simulate_event_slow(0.0, [0.0, 0.0, 0.0]) == 0
    assert vectorized.simulate_event_fast(0.0, [0.0, 0.0, 0.0]) == 0
    assert isinstance(reference.simulate_event_slow(0.01, [0, 0, 0]), int)
    assert isinstance(vectorized.simulate_event_fast(0.01, [0, 0, 0]), int)
    print('Проверки геометрии и методов пройдены.')


def compare_speed(energy=5.0, iterations=100, position=(0.3, -0.2, 0.8), seed=42):
    if energy < 0 or iterations <= 0:
        raise ValueError('Энергия должна быть неотрицательной, число событий — положительным')

    reference = SlowDetectorAndSource(seed=seed)
    vectorized = FastDetectorAndSource(seed=seed + 1)
    print(f'Сравнение: {iterations} событий по {energy} МэВ, '
          f'примерно {int(round(energy * reference.Y))} фотонов на событие.')

    begin = time.perf_counter()
    reference_signals = np.array([
        reference.simulate_event_slow(energy, position)
        for _ in range(iterations)
    ], dtype=np.int32)
    elapsed_slow = time.perf_counter() - begin

    begin = time.perf_counter()
    vector_signals = np.array([
        vectorized.simulate_event_fast(energy, position)
        for _ in range(iterations)
    ], dtype=np.int32)
    elapsed_fast = time.perf_counter() - begin

    print(f'Циклы Python: {elapsed_slow:.4f} с, средний заряд {reference_signals.mean():.2f} ф.э.')
    print(f'NumPy-массивы: {elapsed_fast:.4f} с, средний заряд {vector_signals.mean():.2f} ф.э.')
    print(f'Ускорение: {elapsed_slow / elapsed_fast:.2f} раза')
    print('Различие отдельных зарядов ожидаемо из-за случайного характера Монте-Карло.')
    return elapsed_slow, elapsed_fast


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Задача 7: векторизация фотонного детектора')
    parser.add_argument('--energy', type=float, default=5.0, help='Энергия события в МэВ')
    parser.add_argument('--events', type=int, default=100, help='Число событий')
    parser.add_argument('--seed', type=int, default=42, help='Зерно генератора')
    options = parser.parse_args()

    check_implementation()
    compare_speed(energy=options.energy, iterations=options.events, seed=options.seed)
