import numpy as np


class Detector:
    def __init__(self, R=2.0, N=40, r=0.1, a=0.5):
        self.R = float(R)
        self.N = int(N)
        self.r = float(r)
        self.a = float(a)

        self.pmt_coordinates = self._generate_uniform_pmts()
        self.pmt_normals = self.pmt_coordinates / self.R

    def _generate_uniform_pmts(self):
        numbers = np.arange(self.N, dtype=float)
        heights = 1.0 - 2.0 * (numbers + 0.5) / self.N
        circle_radii = np.sqrt(1.0 - heights ** 2)

        angle_step = np.pi * (3.0 - np.sqrt(5.0))
        angles = numbers * angle_step
        positions = np.column_stack(
            (
                circle_radii * np.cos(angles),
                heights,
                circle_radii * np.sin(angles),
            )
        )
        return self.R * positions

    def find_pmt_for_point(self, point, eps=1e-6):
        surface_point = np.asarray(point, dtype=float)
        if abs(np.linalg.norm(surface_point) - self.R) > eps:
            return None

        along_normals = self.pmt_normals @ surface_point
        tangent_parts = surface_point[None, :] - (
            along_normals[:, None] * self.pmt_normals
        )
        offsets = np.linalg.norm(tangent_parts, axis=1)

        suitable = (along_normals > 0) & (offsets <= self.r)
        candidates = np.flatnonzero(suitable)
        if candidates.size == 0:
            return None

        chosen = candidates[np.argmin(offsets[candidates])]
        return int(chosen)

    def register_photon(self, point, rng=None):
        hit_index = self.find_pmt_for_point(point)
        if hit_index is None:
            return None

        if rng is None:
            rng = np.random.default_rng()

        is_detected = bool(rng.random() < self.a)
        return hit_index, is_detected


def test_detector():
    model = Detector()

    assert model.pmt_coordinates.shape == (40, 3)
    norms = np.linalg.norm(model.pmt_coordinates, axis=1)
    assert np.allclose(norms, model.R)

    first_pmt = model.pmt_coordinates[0]
    assert model.find_pmt_for_point(first_pmt) == 0
    assert model.find_pmt_for_point([0, 0, 0]) is None
    assert model.find_pmt_for_point([2, 0, 0]) is None

    perfect_model = Detector(a=1.0)
    generator = np.random.default_rng(42)
    response = perfect_model.register_photon(
        perfect_model.pmt_coordinates[0], rng=generator
    )
    assert response == (0, True)

    print("Все тесты пройдены успешно")


if __name__ == "__main__":
    detector = Detector()

    print("Параметры детектора:")
    print(f"R = {detector.R} м")
    print(f"N = {detector.N}")
    print(f"r = {detector.r} м")
    print(f"a = {detector.a}")

    print("\nПервые 3 ФЭУ:")
    print(detector.pmt_coordinates[:3])

    test_detector()
