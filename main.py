import MOEAD
import NSGA2
import numpy as np
import matplotlib.pyplot as plt
import Plot
import pickle
import copy
import sys
from tqdm import tqdm
# from pycallgraph2 import PyCallGraph
# from pycallgraph2.output import GraphvizOutput
import argparse
import yaml

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    pop_size = config['POP_SIZE']
    neighborhood_size = config['NEIGHBORHOOD_SIZE']
    num_sensors = config['NUM_SENSORS']
    num_sink_nodes = config['NUM_SINK_NODES']
    num_generations = config['NUM_GENERATION']
    length = config['DIMENSIONS']['LENGTH']
    width = config['DIMENSIONS']['WIDTH']
    cr = config['OPERATORS']['CROSSOVER RATE']
    mr = config['OPERATORS']['MUTATION RATE']

    # parser = argparse.ArgumentParser()
    # Take in argument as epoch number for saving result file when run via bash script, default is 0
    if (len(sys.argv) > 1):
        epoch = sys.argv[1]
        dataset_no = sys.argv[2]
    else:
        epoch = 0
        # Load positions
        dataset_no = 0
    with open(f'Datasets/uniform/{width}x{length}unit/{num_sensors}sensors/sensors_positions_{dataset_no}.pickle', 'rb') as file:
        sensors_positions = pickle.load(file)
    with open('Datasets/sink_nodes_positions.pickle', 'rb') as file:
        sink_nodes_positions = pickle.load(file)

    # Run
    # population = NSGA2.Population(pop_size,num_sensors,sensors_positions,num_sink_nodes,sink_nodes_positions)
    population = MOEAD.Population(
        pop_size,
        neighborhood_size=neighborhood_size,
        num_sensors=num_sensors,
        sensors_positions=sensors_positions,
        num_sink_nodes=num_sink_nodes,
        sink_nodes_positions=sink_nodes_positions,
        barrier_length=length,
        crossover_rate=cr,
        mutation_rate=mr)

    objectives_by_generations = []
    first_solutions = [indi.solution for indi in population.pop]
    with open(f'MOEAD_Results/uniform/{width}x{length}unit/{num_sensors}sensors/dataset_{dataset_no}/first_solutions_{epoch}.pickle', 'wb') as file:
        pickle.dump(first_solutions, file)

    for i in tqdm(range(num_generations)):
        population.reproduct()
        f = []
        fitnesses = []
        best_fitness = 0
        for indi in population.pop:
            f.append(copy.deepcopy(indi.f))

        objectives_by_generations.append(f)

    last_solutions = [indi.solution for indi in population.pop]

    # Change file name everytime!
    with open(f'MOEAD_Results/uniform/{width}x{length}unit/{num_sensors}sensors/dataset_{dataset_no}/last_solutions_{epoch}.pickle', 'wb') as file:
        pickle.dump(last_solutions, file)
    with open(f'MOEAD_Results/uniform/{width}x{length}unit/{num_sensors}sensors/dataset_{dataset_no}/objectives_by_generations_{epoch}.pickle', 'wb') as file:
        pickle.dump(objectives_by_generations, file)
    with open(f'MOEAD_Results/uniform/{width}x{length}unit/{num_sensors}sensors/dataset_{dataset_no}/lambdas_{epoch}.pickle', 'wb') as file:
        pickle.dump(population.lambdas, file)
    with open(f'MOEAD_Results/uniform/{width}x{length}unit/{num_sensors}sensors/dataset_{dataset_no}/ep_{epoch}.pickle', 'wb') as file:
        pickle.dump([indi.solution for indi in population.EP], file)

    print('Finished!')
