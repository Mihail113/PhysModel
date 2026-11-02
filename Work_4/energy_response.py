import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class Source:
    def __init__(self, Y=10000):
        self.Y = Y

    def generate_photons(self, E):
        photon_count = int(round(self.Y * E))
        polar_cosines = np.random.uniform(-1.0, 1.0, photon_count)
        azimuths = np.random.uniform(0.0, 2.0 * np.pi, photon_count)
        transverse = np.sqrt(1.0 - polar_cosines ** 2)

        components = (
            transverse * np.cos(azimuths),
            transverse * np.sin(azimuths),
            polar_cosines,
        )
        return np.column_stack(components).astype(np.float32)


class Detector:
    def __init__(self, R=2.0, N=40, r=0.1, a=0.5):
        self.R = R
        self.N = N
        self.r = r
        self.a = a
        self.pmt_coordinates = self._generate_uniform_pmts()

    def _generate_uniform_pmts(self):
        numbers = np.arange(self.N, dtype=float)
        elevations = 1.0 - 2.0 * numbers / (self.N - 1)
        ring_sizes = np.sqrt(1.0 - elevations ** 2)
        turns = np.pi * (np.sqrt(5.0) - 1.0) * numbers

        coordinates = np.column_stack((
            ring_sizes * np.cos(turns),
            elevations,
            ring_sizes * np.sin(turns),
        ))
        return (self.R * coordinates).astype(np.float32)

    def find_pmt_for_points_vectorized(self, hit_points):
        if len(hit_points) == 0:
            return np.empty(0, dtype=np.int32)

        # Все ФЭУ расположены на одной сфере, поэтому ближайший
        # центр соответствует максимальному скалярному произведению.
        projections = hit_points @ self.pmt_coordinates.T
        nearest = np.argmax(projections, axis=1)
        offsets = hit_points - self.pmt_coordinates[nearest]
        squared_distances = np.einsum('ij,ij->i', offsets, offsets)

        return np.where(squared_distances <= self.r ** 2, nearest, -1)

    def simulate_event(self, source_pos, directions):
        if len(directions) == 0:
            return 0

        emission_point = np.asarray(source_pos, dtype=np.float32)
        linear_term = 2.0 * (directions @ emission_point)
        constant_term = np.dot(emission_point, emission_point) - self.R ** 2
        discriminants = linear_term ** 2 - 4.0 * constant_term

        possible = discriminants >= 0.0
        flight_distances = (-linear_term + np.sqrt(np.maximum(discriminants, 0.0))) / 2.0
        forward = possible & (flight_distances >= 0.0)
        contact_points = emission_point + directions[forward] * flight_distances[forward, None]
        matched_pmts = self.find_pmt_for_points_vectorized(contact_points)
        hits_count = np.count_nonzero(matched_pmts >= 0)

        if hits_count == 0:
            return 0

        # Один попавший фотон порождает пуассоновское число фотоэлектронов.
        return int(np.random.poisson(lam=self.a, size=hits_count).sum())


def collect_charges(detector, source, energy, origin, event_count):
    samples = np.empty(event_count, dtype=np.int32)
    for event_number in range(event_count):
        photon_vectors = source.generate_photons(energy)
        samples[event_number] = detector.simulate_event(origin, photon_vectors)
    return samples


def main():
    np.random.seed(42)
    detector = Detector(R=2.0, N=40, r=0.1, a=0.5)
    source = Source(Y=10000)
    event_count = 1000
    center = [0.0, 0.0, 0.0]
    displaced = [0.0, 0.0, 1.0]

    print('Моделирование 1000 событий при E = 1 МэВ в центре...')
    center_signals = collect_charges(detector, source, 1.0, center, event_count)

    energy_values = np.array([0.2, 0.5, 0.7, 1.5, 2.0, 3.0])
    relative_widths = []
    print('\nЗависимость ширины распределения от энергии:')
    for energy in energy_values:
        measured = collect_charges(detector, source, energy, center, event_count)
        average = np.mean(measured)
        spread = np.std(measured)
        relative_widths.append(spread / average)
        print(f'E = {energy:>3.1f} МэВ | среднее = {average:>7.2f} | '
              f'СКО = {spread:>6.2f} | σ/μ = {spread / average:.4f}')

    print('\nМоделирование 1000 событий при смещении z = 1 м...')
    displaced_signals = collect_charges(detector, source, 1.0, displaced, event_count)

    average_center = np.mean(center_signals)
    average_displaced = np.mean(displaced_signals)
    percent_change = (average_displaced / average_center - 1.0) * 100.0
    print(f'Средний заряд в центре: {average_center:.2f} ф.э.')
    print(f'Средний заряд при z = 1 м: {average_displaced:.2f} ф.э.')
    print(f'Изменение среднего: {percent_change:+.2f}%')

    # Для источника в центре p = N*r^2/(4*R^2).
    # С учетом пуассоновского числа фотоэлектронов на один фотон:
    # Var(Q) = n*p*a*(1 + a*(1-p)).
    photon_hit_probability = detector.N * detector.r ** 2 / (4.0 * detector.R ** 2)
    theoretical_factor = np.sqrt(
        (1.0 + detector.a * (1.0 - photon_hit_probability))
        / (source.Y * photon_hit_probability * detector.a)
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].hist(center_signals, bins=25, alpha=0.65, label='Центр', edgecolor='black')
    axes[0].hist(displaced_signals, bins=25, alpha=0.55, label='z = 1 м', edgecolor='black')
    axes[0].set(title='Заряд для E = 1 МэВ', xlabel='Заряд (ф.э.)', ylabel='Число событий')
    axes[0].legend()

    axes[1].plot(energy_values, relative_widths, 'o-', linewidth=2)
    axes[1].set(title='Относительная ширина', xlabel='Энергия (МэВ)', ylabel='σ/μ')

    smooth_energies = np.linspace(0.1, 3.2, 150)
    axes[2].plot(energy_values, relative_widths, 'o', label='Монте-Карло')
    axes[2].plot(smooth_energies, theoretical_factor / np.sqrt(smooth_energies),
                 '--', label=r'Теория $\propto 1/\sqrt{E}$')
    axes[2].set(title='Проверка закона 1/√E', xlabel='Энергия (МэВ)', ylabel='σ/μ')
    axes[2].legend()

    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    chart_path = Path(__file__).with_name('energy_response.png')
    fig.savefig(chart_path, dpi=150)
    print(f'Графики сохранены: {chart_path}')
    plt.show()


if __name__ == '__main__':
    main()
