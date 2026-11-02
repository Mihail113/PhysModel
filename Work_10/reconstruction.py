"""Работа 10: байесовская реконструкция энергии и координат сцинтилляции."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Detector:
    def __init__(self, R=2.0, N=40, r=0.1, a=0.5, Y=10000,
                 attenuation_length=8.0):
        self.R = float(R)
        self.N = int(N)
        self.r = float(r)
        self.a = float(a)
        self.Y = int(Y)
        self.attenuation_length = float(attenuation_length)
        self.pmt_coordinates = self._make_pmt_coordinates()

    def _make_pmt_coordinates(self):
        sensor_numbers = np.arange(self.N, dtype=float)
        vertical_levels = 1.0 - 2.0 * sensor_numbers / (self.N - 1)
        ring_radii = np.sqrt(1.0 - vertical_levels**2)
        spiral_phases = np.pi * (np.sqrt(5.0) - 1.0) * sensor_numbers
        points = np.column_stack((
            ring_radii * np.cos(spiral_phases),
            vertical_levels,
            ring_radii * np.sin(spiral_phases),
        ))
        return self.R * points

    def expected_charges(self, event_energy, event_position):
        """Средние заряды всех ФЭУ в приближении малых фотокатодов."""
        distances = np.linalg.norm(
            self.pmt_coordinates - np.asarray(event_position), axis=1
        )
        geometric_factor = self.r**2 / (4.0 * distances**2)
        survival_factor = np.exp(-distances / self.attenuation_length)
        return event_energy * self.Y * geometric_factor * survival_factor * self.a

    def simulate_event(self, energy, position, rng):
        return rng.poisson(self.expected_charges(energy, position))


def construct_pymc_model(detector, measured_charges):
    import pymc as pm

    with pm.Model() as response_model:
        sensor_positions = pm.Data("sensor_positions", detector.pmt_coordinates)
        observed_counts = pm.Data("measured_charges", measured_charges)

        estimated_energy = pm.Uniform("Energy", lower=0.1, upper=5.0)
        coord_x = pm.Uniform("X", lower=-detector.R, upper=detector.R)
        coord_y = pm.Uniform("Y", lower=-detector.R, upper=detector.R)
        coord_z = pm.Uniform("Z", lower=-detector.R, upper=detector.R)
        estimated_position = pm.math.stack([coord_x, coord_y, coord_z])

        # Источник должен находиться внутри объёма детектора, а не просто куба.
        radius_squared = pm.math.sum(estimated_position**2)
        pm.Potential(
            "inside_detector",
            pm.math.switch(radius_squared < detector.R**2, 0.0, -np.inf),
        )

        distances_squared = pm.math.sum(
            (sensor_positions - estimated_position) ** 2, axis=1
        )
        # Небольшой численный порог предотвращает деление на ноль.
        safe_distances = pm.math.sqrt(
            pm.math.maximum(distances_squared, (detector.r / 10.0) ** 2)
        )
        mean_response = (
            estimated_energy * detector.Y
            * detector.r**2 / (4.0 * safe_distances**2)
            * pm.math.exp(-safe_distances / detector.attenuation_length)
            * detector.a
        )
        pm.Poisson("PMT_Charge", mu=mean_response, observed=observed_counts)

    return response_model


def draw_posteriors(trace, true_energy, true_position, save_path):
    labels = ("Energy", "X", "Y", "Z")
    reference_values = (true_energy, *true_position)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    for axis, parameter, exact_value in zip(axes.flat, labels, reference_values):
        sampled_values = np.asarray(trace.posterior[parameter]).reshape(-1)
        average = sampled_values.mean()
        axis.hist(sampled_values, bins=35, density=True, alpha=0.7,
                  edgecolor="black", label="Апостериорное распределение")
        axis.axvline(exact_value, color="red", linestyle="--", linewidth=2,
                     label=f"Истинное: {exact_value:.2f}")
        axis.axvline(average, color="black", linestyle=":", linewidth=2,
                     label=f"Среднее: {average:.2f}")
        axis.set(title=f"Реконструкция: {parameter}",
                 xlabel="МэВ" if parameter == "Energy" else "м",
                 ylabel="Плотность вероятности")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def validate_geometry(detector, energy, position):
    assert detector.pmt_coordinates.shape == (40, 3)
    assert np.allclose(np.linalg.norm(detector.pmt_coordinates, axis=1), detector.R)
    assert np.linalg.norm(position) < detector.R
    assert np.all(detector.expected_charges(energy, position) > 0.0)
    assert detector.simulate_event(energy, position, np.random.default_rng(19)).shape == (40,)
    print("Проверки геометрии и генерации сигналов пройдены.")


def main():
    parser = argparse.ArgumentParser(description="Работа 10: байесовская реконструкция события")
    parser.add_argument("--draws", type=int, default=1500)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.draws, args.tune, args.chains) < 1:
        parser.error("draws, tune и chains должны быть положительными")

    detector = Detector(R=2.0, N=40, r=0.1, a=0.5, Y=10000,
                        attenuation_length=8.0)
    true_energy = 1.35
    true_position = np.array([0.4, -0.3, 0.7])
    validate_geometry(detector, true_energy, true_position)

    random_generator = np.random.default_rng(args.seed)
    measured = detector.simulate_event(true_energy, true_position, random_generator)
    print(f"Истинная энергия: {true_energy:.2f} МэВ")
    print(f"Истинная позиция: {true_position} м")
    print(f"Заряд: {measured.sum()} ф.э.; сработало {np.count_nonzero(measured)} ФЭУ из {detector.N}")

    import pymc as pm
    model = construct_pymc_model(detector, measured)
    with model:
        posterior = pm.sample(
            draws=args.draws, tune=args.tune, chains=args.chains, cores=1,
            target_accept=0.9, random_seed=args.seed,
            initvals={"Energy": 1.0, "X": 0.0, "Y": 0.0, "Z": 0.0},
            return_inferencedata=True,
        )

    print("\nРезультаты байесовской реконструкции:")
    truth = {"Energy": true_energy, "X": true_position[0],
             "Y": true_position[1], "Z": true_position[2]}
    for variable, actual in truth.items():
        samples = np.asarray(posterior.posterior[variable]).reshape(-1)
        print(f"{variable:>6}: истина {actual:>6.3f}; "
              f"оценка {samples.mean():>6.3f} ± {samples.std():.3f}")

    output_file = Path(__file__).with_name("reconstruction.png")
    draw_posteriors(posterior, true_energy, true_position, output_file)
    print(f"Графики сохранены: {output_file}")


if __name__ == "__main__":
    main()
