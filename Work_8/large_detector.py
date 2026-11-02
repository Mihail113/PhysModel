import time
import numpy as np
from scipy.spatial import cKDTree


class LargeDetector:
    def __init__(self, R=15.0, N=2000, r=0.1, a=0.5):
        self.R = float(R)
        self.N = int(N)
        self.r = float(r)
        self.a = float(a)
        self.pmt_coordinates = self._generate_uniform_pmts()
        self.spatial_index = cKDTree(self.pmt_coordinates)

    def _generate_uniform_pmts(self):
        sensor_ids = np.arange(self.N, dtype=np.float64)
        latitudes = 1.0 - 2.0 * sensor_ids / (self.N - 1)
        ring_radii = np.sqrt(1.0 - latitudes ** 2)
        spiral_angles = np.pi * (np.sqrt(5.0) - 1.0) * sensor_ids

        positions = np.column_stack((
            ring_radii * np.cos(spiral_angles),
            latitudes,
            ring_radii * np.sin(spiral_angles),
        ))
        return (self.R * positions).astype(np.float32)

    def find_pmt_with_kdtree(self, hit_points):
        separations, nearest_ids = self.spatial_index.query(
            np.asarray(hit_points), k=1, distance_upper_bound=self.r
        )
        return np.where(np.isfinite(separations), nearest_ids, -1).astype(np.int32)

    def find_pmt_with_full_scan(self, hit_points, batch_size=1000):
        impacts = np.asarray(hit_points, dtype=np.float32)
        matches = np.full(len(impacts), -1, dtype=np.int32)

        for left in range(0, len(impacts), batch_size):
            right = min(left + batch_size, len(impacts))
            differences = impacts[left:right, None, :] - self.pmt_coordinates[None, :, :]
            squared_distances = np.einsum('ijk,ijk->ij', differences, differences)
            closest = np.argmin(squared_distances, axis=1)
            closest_dist_sq = squared_distances[np.arange(right - left), closest]
            matches[left:right] = np.where(closest_dist_sq <= self.r ** 2, closest, -1)

        return matches

    def calculate_hit_points(self, source_pos, directions):
        origin = np.asarray(source_pos, dtype=np.float32)
        rays = np.asarray(directions, dtype=np.float32)
        linear_part = 2.0 * (rays @ origin)
        free_part = np.dot(origin, origin) - self.R ** 2
        discriminant = linear_part ** 2 - 4.0 * free_part
        distances = (-linear_part + np.sqrt(np.maximum(discriminant, 0.0))) / 2.0
        return origin + rays * distances[:, None]

    def simulate_event(self, pmt_indices):
        hit_count = np.count_nonzero(np.asarray(pmt_indices) >= 0)
        return int(np.random.poisson(self.a, size=hit_count).sum())


def main():
    np.random.seed(42)
    detector = LargeDetector(R=15.0, N=2000, r=0.1, a=0.5)
    event_energy = 10.0
    emitted_count = int(event_energy * 10000)
    source_position = [2.0, -3.0, 5.0]

    cos_polar = np.random.uniform(-1.0, 1.0, emitted_count)
    azimuth = np.random.uniform(0.0, 2.0 * np.pi, emitted_count)
    transverse = np.sqrt(1.0 - cos_polar ** 2)
    photon_directions = np.column_stack((
        transverse * np.cos(azimuth),
        transverse * np.sin(azimuth),
        cos_polar,
    )).astype(np.float32)
    impact_points = detector.calculate_hit_points(source_position, photon_directions)

    print('--- Параметры большого детектора ---')
    print(f'R = {detector.R} м, N = {detector.N}, r = {detector.r} м, a = {detector.a}')
    print(f'Энергия: {event_energy} МэВ; фотонов: {emitted_count}')

    start_time = time.perf_counter()
    brute_force_hits = detector.find_pmt_with_full_scan(impact_points)
    brute_force_seconds = time.perf_counter() - start_time

    start_time = time.perf_counter()
    tree_hits = detector.find_pmt_with_kdtree(impact_points)
    tree_seconds = time.perf_counter() - start_time

    matches = np.array_equal(brute_force_hits, tree_hits)
    print(f'Полный перебор: {brute_force_seconds:.4f} с; попаданий: {np.count_nonzero(brute_force_hits >= 0)}')
    print(f'KD-Tree:        {tree_seconds:.4f} с; попаданий: {np.count_nonzero(tree_hits >= 0)}')
    print(f'Результаты совпадают: {matches}')
    if not matches:
        raise AssertionError('Методы поиска дали разные результаты')

    print(f'Собранный заряд: {detector.simulate_event(tree_hits)} ф.э.')
    print(f'Ускорение: {brute_force_seconds / tree_seconds:.1f} раз')


if __name__ == '__main__':
    main()
