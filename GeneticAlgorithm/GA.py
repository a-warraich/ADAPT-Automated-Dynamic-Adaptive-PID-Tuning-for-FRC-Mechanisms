import random
import time
from typing import Tuple, List, Callable

import ntcore
import numpy as np
import matplotlib.pyplot as plt


# ================== Genetic Algorithm ==================

class GeneticAlgorithm:
    def __init__(
        self,
        population_size: int,
        kp_bounds: Tuple[float, float],
        ki_bounds: Tuple[float, float],
        kd_bounds: Tuple[float, float],
        generations: int,
        mutation_rate: float,
        desiredState: float,
    ):
        self.population_size = population_size
        self.kp_bounds = kp_bounds
        self.ki_bounds = ki_bounds
        self.kd_bounds = kd_bounds
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.desiredState = desiredState
        self.population = self._create_initial_population()

    def _create_initial_population(self) -> List[Tuple[float, float, float]]:
        pop = []
        for _ in range(self.population_size):
            kp = random.uniform(*self.kp_bounds)
            ki = random.uniform(*self.ki_bounds)
            kd = random.uniform(*self.kd_bounds)
            pop.append((kp, ki, kd))
        return pop

    def fitness_function(
        self,
        error_list: List[float],
        output_list: List[float],
        desired_value: float,
    ) -> float:
        if not error_list:
            return -1e9

        iae = sum(abs(e) for e in error_list)
        overshoot = 0.0
        if output_list:
            mx = max(output_list)
            if mx > desired_value:
                overshoot = mx - desired_value

        return -(iae + 10.0 * overshoot)

    def _mutate(self, individual: Tuple[float, float, float]) -> Tuple[float, float, float]:
        ind = list(individual)
        bounds = [self.kp_bounds, self.ki_bounds, self.kd_bounds]
        for i in range(3):
            if random.random() < self.mutation_rate:
                lo, hi = bounds[i]
                span = hi - lo
                delta = random.uniform(-0.1 * span, 0.1 * span)
                ind[i] = max(lo, min(hi, ind[i] + delta))
        return tuple(ind)

    def _crossover(self, p1, p2):
        alpha = random.random()
        c1 = tuple(alpha * a + (1 - alpha) * b for a, b in zip(p1, p2))
        c2 = tuple(alpha * b + (1 - alpha) * a for a, b in zip(p1, p2))
        return c1, c2

    def _selection(self, population, fitnesses, tournament_size=3):
        selected = []
        pairs = list(zip(population, fitnesses))
        for _ in range(len(population)):
            t = random.sample(pairs, k=tournament_size)
            best = max(t, key=lambda x: x[1])[0]
            selected.append(best)
        return selected

    def run(
        self,
        external_simulator: Callable[[Tuple[float, float, float]], Tuple[List[float], List[float], float]],
    ) -> Tuple[float, float, float]:
        pop = self.population
        for gen in range(self.generations):
            print(f"\n=== Generation {gen+1}/{self.generations} ===")
            fits = []
            for i, ind in enumerate(pop):
                print(f"  Individual {i+1}/{len(pop)}: {ind}")
                errors, outputs, desired = external_simulator(ind)
                fit = self.fitness_function(errors, outputs, desired)
                fits.append(fit)
                print(f"    fitness = {fit:.4f}")

            best_idx = fits.index(max(fits))
            best = pop[best_idx]
            print(f"  -> Best this gen: {best} (fitness={fits[best_idx]:.4f})")

            selected = self._selection(pop, fits)
            next_pop = []
            for i in range(0, len(selected) - 1, 2):
                c1, c2 = self._crossover(selected[i], selected[i + 1])
                next_pop.append(self._mutate(c1))
                next_pop.append(self._mutate(c2))
            if len(selected) % 2 == 1:
                next_pop.append(selected[-1])

            next_pop[0] = best
            pop = next_pop

        final_fits = []
        for ind in pop:
            errors, outputs, desired = external_simulator(ind)
            final_fits.append(self.fitness_function(errors, outputs, desired))

        best_idx = final_fits.index(max(final_fits))
        best = pop[best_idx]
        print(f"\n=== Overall Best PID: {best}, fitness={final_fits[best_idx]:.4f} ===")
        return best


# ================== Plotting ==================

def plot_response_curve(outputs, desired_rpm, sample_rate_hz=50.0, title="PID Response Curve"):
    n = len(outputs)
    if n == 0:
        print("No samples to plot.")
        return

    t = np.arange(n) / sample_rate_hz

    plt.figure(figsize=(10, 5))
    plt.plot(t, outputs, label="Measured RPM", linewidth=2)
    plt.axhline(y=desired_rpm, linestyle="--", label="Setpoint", linewidth=2)

    plt.xlabel("Time (s)")
    plt.ylabel("RPM")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ================== Robot NT Interface (NT4 via ntcore) ==================

class RobotExternalSimulator:

    def __init__(self, desired_rpm: float, team: int, use_sim: bool = True):
        self.desired_rpm = float(desired_rpm)

        self.inst = ntcore.NetworkTableInstance.getDefault()
        self.inst.startClient4("GA_PID_Client")

        if use_sim:
            self.inst.setServer("127.0.0.1")
        else:
            self.inst.setServerTeam(team)

        self.inst.startDSClient()

        print("Waiting for NT connection...")
        time.sleep(1.0)
        print("Connected? ", self.inst.isConnected())

        table = self.inst.getTable("GA_PID")

        self.kp_pub = table.getDoubleTopic("kp").publish()
        self.ki_pub = table.getDoubleTopic("ki").publish()
        self.kd_pub = table.getDoubleTopic("kd").publish()
        self.setpoint_pub = table.getDoubleTopic("setpoint").publish()
        self.start_pub = table.getBooleanTopic("startTest").publish()

        self.errors_sub = table.getDoubleArrayTopic("errors").subscribe([])
        self.outputs_sub = table.getDoubleArrayTopic("outputs").subscribe([])
        self.done_sub = table.getBooleanTopic("testDone").subscribe(False)

        self.min_samples = 50
        self.max_wait_s = 6.0

    def __call__(self, ind):
        return self.evaluate_individual(ind)

    def evaluate_individual(self, ind):
        kp, ki, kd = ind
        print(f"  -> Sending to robot: Kp={kp}, Ki={ki}, Kd={kd}")

        self.start_pub.set(False)
        self.errors_sub.get()
        self.outputs_sub.get()
        self.done_sub.get()

        self.kp_pub.set(float(kp))
        self.ki_pub.set(float(ki))
        self.kd_pub.set(float(kd))
        self.setpoint_pub.set(self.desired_rpm)

        self.start_pub.set(True)
        self.inst.flush()

        start_time = time.time()
        last_len = 0

        while True:
            errors = list(self.errors_sub.get())
            outputs = list(self.outputs_sub.get())
            cur_len = min(len(errors), len(outputs))
            elapsed = time.time() - start_time

            if cur_len >= self.min_samples:
                break
            if self.done_sub.get():
                break
            if elapsed > self.max_wait_s:
                print("  WARNING: Robot timeout.")
                break

            if cur_len != last_len:
                print(f"    samples so far: {cur_len}")
                last_len = cur_len

            time.sleep(0.02)

        self.start_pub.set(False)

        errors = list(self.errors_sub.get())
        outputs = list(self.outputs_sub.get())

        if not errors or not outputs:
            print("  ERROR: No data received from robot.")
            errors = [1000.0]
            outputs = [0.0]
        else:
            print(f"  <- Received {len(errors)} samples")

        return errors, outputs, self.desired_rpm


# ================== Main ==================

def main():
    TEAM_NUMBER = 3624
    DESIRED_RPM = 2000.0

    ga = GeneticAlgorithm(
        population_size=8,
        kp_bounds=(0.0005, 0.05),
        ki_bounds=(0.0, 0.01),
        kd_bounds=(0.0, 0.01),
        generations=3,
        mutation_rate=0.2,
        desiredState=DESIRED_RPM,
    )

    robot_sim = RobotExternalSimulator(desired_rpm=DESIRED_RPM, team=TEAM_NUMBER, use_sim=True)
    best = ga.run(robot_sim)
    print("\nFINAL BEST PID:", best)

    errors, outputs, desired = robot_sim(best)
    plot_response_curve(
        outputs,
        desired,
        sample_rate_hz=50.0,
        title=f"Final Best PID Response (Kp={best[0]:.4f}, Ki={best[1]:.4f}, Kd={best[2]:.4f})",
    )


if __name__ == "__main__":
    main()
