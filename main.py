import MOEAD
import NSGA2
import numpy as np
import matplotlib.pyplot as plt
import Plot
import pickle
import copy
import sys
from tqdm import tqdm
from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput
import argparse
import yaml

def run_MOEAD():
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
    with open(f'MOEAD_Results/{distr}/{width}x{length}unit/{num_sensors}sensors/dataset_{dataset_no}/first_solutions_{epoch}.pickle', 'wb') as file:
        pickle.dump(first_solutions, file)

    print('Running algorithm...')
    for i in tqdm(range(num_generations)):
        population.reproduct()
        f = []
        fitnesses = []
        best_fitness = 0
        for indi in population.pop:
            f.append(copy.deepcopy(indi.f))

        objectives_by_generations.append(f)

    last_solutions = [indi.solution for indi in population.pop]

    # Save results
    print('Saving results...')
    with open(f'MOEAD_Results/{distr}/{width}x{length}unit/{num_sensors}sensors/dataset_{dataset_no}/last_solutions_{epoch}.pickle', 'wb') as file:
        pickle.dump(last_solutions, file)
    with open(f'MOEAD_Results/{distr}/{width}x{length}unit/{num_sensors}sensors/dataset_{dataset_no}/objectives_by_generations_{epoch}.pickle', 'wb') as file:
        pickle.dump(objectives_by_generations, file)
    with open(f'MOEAD_Results/{distr}/{width}x{length}unit/{num_sensors}sensors/dataset_{dataset_no}/lambdas_{epoch}.pickle', 'wb') as file:
        pickle.dump(population.lambdas, file)
    with open(f'MOEAD_Results/{distr}/{width}x{length}unit/{num_sensors}sensors/dataset_{dataset_no}/ep_{epoch}.pickle', 'wb') as file:
        pickle.dump([indi.solution for indi in population.EP], file)

    print('Finished!')


def run_NSGA2():
    population = NSGA2.Population(
        pop_size,
        num_sensors=num_sensors,
        sensors_positions=sensors_positions,
        num_sink_nodes=num_sink_nodes,
        sink_nodes_positions=sink_nodes_positions,
        barrier_length=length,
        crossoverate=cr,
        mutation_rate=mr
    )

    objectives_by_generations = []
    first_solutions = [indi.solution for indi in population.pop]
    with open(f'NSGA2_Results/{distr}/{width}x{length}unit/{num_sensors}sensors/dataset_{dataset_no}/first_solutions_{epoch}.pickle', 'wb') as file:
        pickle.dump(first_solutions, file)

    print('Running algorithm...')
    for i in tqdm(range(num_generations)):
        population.reproduct()
        f = []
        fitnesses = []
        best_fitness = 0
        for indi in population.pop:
            f.append(copy.deepcopy(indi.f))

        objectives_by_generations.append(f)

    last_solutions = [indi.solution for indi in population.pop]

    # Save results
    print('Saving results...')
    with open(f'NSGA2_Results/{distr}/{width}x{length}unit/{num_sensors}sensors/dataset_{dataset_no}/last_solutions_{epoch}.pickle', 'wb') as file:
        pickle.dump(last_solutions, file)
    with open(f'NSGA2_Results/{distr}/{width}x{length}unit/{num_sensors}sensors/dataset_{dataset_no}/objectives_by_generations_{epoch}.pickle', 'wb') as file:
        pickle.dump(objectives_by_generations, file)
    with open(f'NSGA2_Results/{distr}/{width}x{length}unit/{num_sensors}sensors/dataset_{dataset_no}/ep_{epoch}.pickle', 'wb') as file:
        pickle.dump([indi.solution for indi in population.EP], file)

    print('Finished!')

if __name__ == "__main__":
    print('Loading configurations...')
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    pop_size = config['POP_SIZE']
    neighborhood_size = config['NEIGHBORHOOD_SIZE']
    num_sensors = config['NUM_SENSORS']
    num_sink_nodes = config['NUM_SINK_NODES']
    num_generations = config['NUM_GENERATION']
    length = config['DIMENSIONS']['LENGTH']
    width = config['DIMENSIONS']['WIDTH']
    distr = config['DISTRIBUTION']
    cr = config['OPERATORS']['CROSSOVER RATE']
    mr = config['OPERATORS']['MUTATION RATE']
    print('Completed loading configurations!')

    # parser = argparse.ArgumentParser()
    # Take in argument as epoch number for saving result file when run via bash script, default is 0
    if (len(sys.argv) > 1):
        epoch = sys.argv[1]
        dataset_no = sys.argv[2]
    else:
        epoch = 0
        dataset_no = 0

    print('Loading sensors and sink nodes positions...')
    # Load positions
    with open(f'Datasets/{distr}/{width}x{length}unit/{num_sensors}sensors/sensors_positions_{dataset_no}.pickle', 'rb') as file:
        sensors_positions = pickle.load(file)
    with open('Datasets/sink_nodes_positions.pickle', 'rb') as file:
        sink_nodes_positions = pickle.load(file)
    print('Completed loading positions!')
    # Run
    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'basic.png'
    # with PyCallGraph(output=graphviz):
    #     run_NSGA2()
    # run_NSGA2()
    run_MOEAD()