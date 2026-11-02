import numpy as np


class Source:
    def __init__(self, Y=10000):
        self.Y = Y

    def generate_photons(self, E):
        photon_count = int(self.Y * E)
        cos_polar = np.random.uniform(-1.0, 1.0, photon_count)
        azimuth = np.random.uniform(0.0, 2.0 * np.pi, photon_count)
        transverse = np.sqrt(1.0 - cos_polar ** 2)

        unit_vectors = np.stack(
            (transverse * np.cos(azimuth),
             transverse * np.sin(azimuth),
             cos_polar),
            axis=1,
        )
        return unit_vectors.astype(np.float32)


class Detector:
    def __init__(self, R=2.0, N=40, r=0.1, a=0.5):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
        self.pmts = self.make_pmts()

    def make_pmts(self):
        positions = []
        rotation_step = np.pi * (np.sqrt(5.0) - 1.0)

        for pmt_number in range(self.N):
            height = 1.0 - 2.0 * pmt_number / (self.N - 1)
            circle_radius = np.sqrt(1.0 - height ** 2)
            rotation = rotation_step * pmt_number

            coordinates = (
                self.R * circle_radius * np.cos(rotation),
                self.R * height,
                self.R * circle_radius * np.sin(rotation),
            )
            positions.append(coordinates)

        return np.asarray(positions, dtype=np.float32)

    def find_pmt(self, point):
        separations = np.linalg.norm(self.pmts - point, axis=1)
        closest_pmt = np.argmin(separations)
        return int(closest_pmt) if separations[closest_pmt] <= self.r else None

    def sample_photoelectrons(self):
        return np.random.poisson(lam=self.a)

    def propagate_and_detect(self, source_pos, directions):
        origin = np.asarray(source_pos, dtype=np.float32)
        photoelectron_counts = np.zeros(self.N, dtype=np.int32)

        linear_coeff = 2.0 * (directions @ origin)
        free_term = np.dot(origin, origin) - self.R ** 2
        discriminants = linear_coeff ** 2 - 4.0 * free_term
        flight_lengths = (
            -linear_coeff + np.sqrt(np.maximum(discriminants, 0.0))
        ) / 2.0
        impact_points = origin + flight_lengths[:, None] * directions

        # Only forward intersections with the detector sphere are physical.
        valid_rays = (discriminants >= 0.0) & (flight_lengths >= 0.0)
        for impact in impact_points[valid_rays]:
            pmt_number = self.find_pmt(impact)
            if pmt_number is not None:
                photoelectron_counts[pmt_number] += self.sample_photoelectrons()

        return photoelectron_counts


def test_model():
    light_source = Source()
    sphere = Detector()
    assert sphere.pmts.shape == (40, 3)
    assert np.allclose(np.linalg.norm(sphere.pmts, axis=1), sphere.R)
    assert light_source.generate_photons(0.0).shape == (0, 3)

    rays = light_source.generate_photons(0.1)
    assert rays.shape == (1000, 3)
    assert np.allclose(np.linalg.norm(rays, axis=1), 1.0, atol=1e-6)
    assert sphere.find_pmt(sphere.pmts[0]) == 0
    assert sphere.propagate_and_detect([0, 0, 0], rays).shape == (40,)
    print("Все тесты пройдены успешно")


if __name__ == "__main__":
    np.random.seed(42)

    R = 2.0
    N = 40
    r = 0.1
    a = 0.5
    Y = 10000
    E = 1.0

    light_source = Source(Y=Y)
    sphere = Detector(R=R, N=N, r=r, a=a)
    emission_point = [0.0, 0.0, 0.0]
    photon_directions = light_source.generate_photons(E)

    pmt_signals = sphere.propagate_and_detect(emission_point, photon_directions)
    measured_charge = int(np.sum(pmt_signals))

    expected_charge = a * Y * E * N * r ** 2 / (4.0 * R ** 2)
    charge_from_task = a * Y * E * N * r ** 2 / R ** 2

    print(f"Энергия источника: {E} МэВ")
    print(f"Количество фотонов: {len(photon_directions)}")
    print(f"Собранный заряд: {measured_charge}")
    print(f"Ожидаемый заряд (по геометрии сферы): {expected_charge:.2f}")
    print(f"Заряд по формуле из условия: {charge_from_task:.2f}")

    test_model()
