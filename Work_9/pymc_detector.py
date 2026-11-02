"""Работа 9. Отклик сферического детектора: PyMC и прямой Монте-Карло.

По умолчанию число испущенных фотонов фиксировано, как в работе 4.
Опция --poisson-source включает пуассоновские флуктуации источника.
Генерация из априорной модели НЕ является MCMC-инференцией.
"""

import argparse
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class Source:
    def __init__(self, Y=10000):
        self.Y = int(Y)

    def photon_count(self, energy):
        if energy < 0:
            raise ValueError("Энергия не может быть отрицательной")
        return int(round(self.Y * energy))


class Detector:
    def __init__(self, R=2.0, N=40, r=0.1, a=0.5, attenuation_length=8.0):
        self.R = float(R)
        self.N = int(N)
        self.r = float(r)
        self.a = float(a)
        self.attenuation_length = float(attenuation_length)
        if self.R <= 0 or self.r <= 0 or self.N <= 0 or self.a < 0:
            raise ValueError("Параметры детектора заданы некорректно")
        if self.attenuation_length <= 0:
            raise ValueError("Длина затухания должна быть положительной")

    def photon_hit_probability(self, attenuation=False):
        # Доля площади ФЭУ на сфере (приближение малых радиусов).
        geometric_fraction = self.N * self.r**2 / (4.0 * self.R**2)
        if not 0.0 <= geometric_fraction <= 1.0:
            raise ValueError("Общая геометрическая вероятность вне [0, 1]")
        transmission_factor = np.exp(-self.R / self.attenuation_length) if attenuation else 1.0
        return geometric_fraction * transmission_factor

    def response_model(self, source, energy=1.0, attenuation=False,
                       poisson_source=False):
        """Строит графовую модель; итоговый заряд условно пуассоновский.

        Фотоны -> попавшие на ФЭУ фотоны -> фотоэлектроны.
        """
        try:
            import pymc as pm
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Для работы PyMC нужна совместимая пара pymc/arviz. "
                "Попробуйте: pip install 'pymc==5.27.1' 'arviz==0.22.0'"
            ) from exc

        probability = self.photon_hit_probability(attenuation)
        with pm.Model() as response:
            # Работа 4: фиксированное число фотонов; работа 5: Poisson(EY).
            if poisson_source:
                photon_total = pm.Poisson(
                    "Emitted_Photons", mu=energy * source.Y
                )
            else:
                photon_total = source.photon_count(energy)
            pmt_hits = pm.Binomial("PMT_Hits", n=photon_total, p=probability)
            pm.Poisson("Total_Photoelectrons", mu=self.a * pmt_hits)
        return response

    def direct_simulation(self, source, energy, event_count, attenuation=False,
                          poisson_source=False, seed=42):
        """Прямой Монте-Карло с использованием точной биномиальной выборки.

        Изотропия источника в центре позволяет заменить явное прослеживание
        каждой траектории её вероятностью попадания на ФЭУ.
        """
        random_generator = np.random.default_rng(seed)
        pmt_probability = self.photon_hit_probability(attenuation)
        if poisson_source:
            photons_per_event = random_generator.poisson(
                lam=energy * source.Y, size=event_count
            )
        else:
            photons_per_event = np.full(
                event_count, source.photon_count(energy), dtype=np.int64
            )
        photons_on_pmts = random_generator.binomial(
            n=photons_per_event, p=pmt_probability
        )
        return random_generator.poisson(
            lam=self.a * photons_on_pmts
        ).astype(np.int32)

    def moments(self, source, energy=1.0, attenuation=False,
                poisson_source=False):
        hit_chance = self.photon_hit_probability(attenuation)
        mean_emission = energy * source.Y if poisson_source else source.photon_count(energy)
        expected_hits = mean_emission * hit_chance
        hit_variance = expected_hits if poisson_source else expected_hits * (1 - hit_chance)
        mean_charge = self.a * expected_hits
        charge_variance = mean_charge + self.a**2 * hit_variance
        return mean_charge, np.sqrt(charge_variance)


def sample_pymc(model, events, seed):
    import pymc as pm
    options = dict(
        random_seed=seed,
        var_names=["Total_Photoelectrons"],
        return_inferencedata=False,
    )
    # PyMC 5.x и 6.x имеют разные названия аргумента числа выборок.
    if "draws" in inspect.signature(pm.sample_prior_predictive).parameters:
        options["draws"] = events
    else:
        options["samples"] = events
    with model:
        generated = pm.sample_prior_predictive(**options)
    return np.asarray(generated["Total_Photoelectrons"]).reshape(-1)


def show_statistics(label, values):
    print(f"{label:<22} среднее = {values.mean():7.2f} ф.э.; "
          f"СКО = {values.std(ddof=0):6.2f} ф.э.")


def run_comparison(detector, source, energy, events, poisson_source,
                   reference_only=False):
    groups = {}
    for use_loss in [False, True]:
        description = "С затуханием" if use_loss else "Без затухания"
        print(f"\n--- {description} ---")
        theoretical_mean, theoretical_std = detector.moments(
            source, energy, use_loss, poisson_source
        )
        print(f"Теория:                  среднее = {theoretical_mean:.2f}; "
              f"СКО = {theoretical_std:.2f}")

        reference = detector.direct_simulation(
            source, energy, events, use_loss, poisson_source,
            seed=55 if use_loss else 45,
        )
        show_statistics("Прямой Монте-Карло:", reference)

        pymc_samples = None
        if not reference_only:
            probabilistic_model = detector.response_model(
                source, energy, use_loss, poisson_source
            )
            pymc_samples = sample_pymc(probabilistic_model, events,
                                       seed=142 if use_loss else 42)
            show_statistics("Генерация PyMC:", pymc_samples)
        groups[description] = (reference, pymc_samples)
    return groups


def save_comparison_graph(groups, output_path, energy):
    figure, graphs = plt.subplots(1, 2, figsize=(13, 5))
    for graph, (description, (reference, pymc_samples)) in zip(graphs, groups.items()):
        graph.hist(reference, bins=25, density=True, alpha=0.55,
                   label="Прямой Монте-Карло", edgecolor="black")
        if pymc_samples is not None:
            graph.hist(pymc_samples, bins=25, density=True, alpha=0.45,
                       label="PyMC", edgecolor="black")
        graph.set_title(description)
        graph.set_xlabel("Собранный заряд (фотоэлектроны)")
        graph.set_ylabel("Плотность вероятности")
        graph.grid(alpha=0.3, linestyle="--")
        graph.legend()
    figure.suptitle(f"Энергетический отклик сферического детектора, E = {energy:g} МэВ")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"\nГрафики записаны в {output_path}")


def test_physics():
    source = Source(Y=10000)
    detector = Detector(R=2.0, N=40, r=0.1, a=0.5, attenuation_length=8.0)
    assert np.isclose(detector.photon_hit_probability(False), 0.025)
    assert np.isclose(detector.moments(source, 1, False)[0], 125.0)
    assert np.isclose(detector.moments(source, 1, True)[0], 125.0 * np.exp(-0.25))
    assert detector.direct_simulation(source, 0.0, 5).tolist() == [0] * 5
    samples = detector.direct_simulation(source, 1.0, 1000, seed=40)
    assert abs(samples.mean() - 125.0) < 5.0
    print("Проверки геометрии и прямой модели пройдены")


def main():
    parser = argparse.ArgumentParser(description="Работа 9: вероятностный отклик детектора")
    parser.add_argument("--events", type=int, default=1000, help="Число событий")
    parser.add_argument("--energy", type=float, default=1.0, help="Энергия (МэВ)")
    parser.add_argument("--poisson-source", action="store_true",
                        help="Пуассоновские флуктуации числа испущенных фотонов")
    parser.add_argument("--reference-only", action="store_true",
                        help="Только NumPy, без запуска PyMC (для диагностики)")
    args = parser.parse_args()
    if args.events <= 0 or args.energy < 0:
        parser.error("Число событий должно быть положительным, энергия — неотрицательной")

    test_physics()
    light_source = Source(Y=10000)
    spherical_detector = Detector(R=2.0, N=40, r=0.1, a=0.5,
                                  attenuation_length=8.0)
    print(f"Энергия {args.energy} МэВ; событий: {args.events}")
    print("Источник:", "пуассоновский" if args.poisson_source else "фиксированный (работа 4)")
    simulations = run_comparison(
        spherical_detector, light_source, args.energy, args.events,
        args.poisson_source, args.reference_only,
    )
    save_comparison_graph(
        simulations, Path(__file__).with_name("pymc_comparison.png"), args.energy
    )
    print("Примечание: prior predictive sampling не является MCMC-выборкой.")


if __name__ == "__main__":
    main()
