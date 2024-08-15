import os
import numpy as np
import pickle
import copy
import MOEAD
import matplotlib.pyplot as plt

if __name__ == "__main__":
    POP_SIZE = 20
    NEIGHBORHOOD_SIZE = 3
    NUM_SENSORS = 100
    NUM_SINK_NODES = 1
    NUM_GENERATION = 500
    NUM_RUNS = 3
    LENGTH, WIDTH = 1000, 50

    mean_fitness_accumulator = np.zeros(NUM_GENERATION)
    mean_best_fitness_accumulator = np.zeros(NUM_GENERATION)

    for run in range(NUM_RUNS):
        with open(f'Datasets/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/sensors_positions_0.pickle', 'rb') as file:
            sensors_positions = pickle.load(file)
        with open('Datasets/sink_nodes_positions.pickle', 'rb') as file:
            sink_nodes_positions = pickle.load(file)

        pop = MOEAD.Population(POP_SIZE, NEIGHBORHOOD_SIZE, NUM_SENSORS, sensors_positions, NUM_SINK_NODES,
                               sink_nodes_positions)

        fitness = []
        best_fitness = []

        for i in range(NUM_GENERATION):
            pop.reproduct()
            generation_fitness = [indi.fitness for indi in pop.pop]
            best = sorted(pop.pop, key=lambda x: x.fitness)[-1]
            best_fitness.append(best.fitness)
            fitness.append(np.mean(generation_fitness))

            if i % 100 == 0:
                print(
                    f'Run: {run+1}, Generation: {i}, Fitness: {best.fitness}')

        mean_fitness_accumulator += np.array(fitness)
        mean_best_fitness_accumulator += np.array(best_fitness)

    mean_fitness_by_generation = mean_fitness_accumulator / NUM_RUNS
    mean_best_fitness_by_generation = mean_best_fitness_accumulator / NUM_RUNS

    result_dir = f'Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/'
    os.makedirs(result_dir, exist_ok=True)
    np.savetxt(os.path.join(result_dir, 'mean_fitness.txt'), mean_fitness_by_generation)
    np.savetxt(os.path.join(result_dir, 'mean_best_fitness.txt'), mean_best_fitness_by_generation)

    # Plotting
    generations = np.arange(1, NUM_GENERATION + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(generations, mean_fitness_by_generation, label='Mean Fitness')
    plt.plot(generations, mean_best_fitness_by_generation, label='Mean Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Mean Fitness and Mean Best Fitness by Generation')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, 'fitness_plot.png'))
    plt.show()
