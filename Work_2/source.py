import numpy as np


class Source:
    def __init__(self, Y=10000, seed=None):
        self.Y = Y
        self.rng = np.random.default_rng(seed)

    def generate_photons(self, E):
        photon_count = int(round(E * self.Y))

        if photon_count == 0:
            return np.empty((0, 3), dtype=np.float32)

        cos_angles = self.rng.uniform(-1.0, 1.0, size=photon_count)
        azimuths = self.rng.uniform(0.0, 2.0 * np.pi, size=photon_count)
        transverse_part = np.sqrt(1.0 - cos_angles ** 2)

        components_x = transverse_part * np.cos(azimuths)
        components_y = transverse_part * np.sin(azimuths)

        unit_vectors = np.stack(
            (components_x, components_y, cos_angles), axis=1
        )
        return unit_vectors.astype(np.float32)


if __name__ == "__main__":
    light_source = Source(Y=10000, seed=42)

    event_energy = 0.5
    photon_vectors = light_source.generate_photons(event_energy)

    print(f"Энергия события: {event_energy} МэВ")
    print(f"Количество фотонов: {len(photon_vectors)}")
    print("Первые 5 направлений:")
    print(photon_vectors[:5])

    if len(photon_vectors) > 0:
        vector_norms = np.linalg.norm(photon_vectors, axis=1)

        print("Длины первых 5 векторов:")
        print(vector_norms[:5])
        print(
            "Максимальное отклонение длины от 1:",
            np.max(np.abs(vector_norms - 1.0))
        )
