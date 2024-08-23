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

if __name__ == "__main__":
	# graphviz = GraphvizOutput()
	# graphviz.output_file = 'basic.png'

	# with PyCallGraph(output=graphviz):
	POP_SIZE = 100
	NEIGHBORHOOD_SIZE = 3
	NUM_SENSORS = 300
	NUM_SINK_NODES = 1
	NUM_GENERATION = 100
	LENGTH, WIDTH = 1000, 50

	# Take in argument as epoch number for saving result file when run via bash script, default is 0
	if(len(sys.argv)>1):
		epoch = sys.argv[1]
		dataset_no = sys.argv[2]
	else:
		epoch = 0
		# Load positions
		dataset_no = 0
	with open(f'Datasets/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/sensors_positions_{dataset_no}.pickle','rb') as file:
		sensors_positions = pickle.load(file)
	with open('Datasets/sink_nodes_positions.pickle','rb') as file:
		sink_nodes_positions = pickle.load(file)
	sink_nodes_positions = [[100, -100]]
	
	# Run
	# population = NSGA2.Population(POP_SIZE,NUM_SENSORS,sensors_positions,NUM_SINK_NODES,sink_nodes_positions)
	population = MOEAD.Population(
		POP_SIZE, 
		neighborhood_size=3, 
		num_sensors=NUM_SENSORS,
		sensors_positions=sensors_positions,
		num_sink_nodes=NUM_SINK_NODES,
		sink_nodes_positions=sink_nodes_positions)

	objectives_by_generations = []
	first_solutions = [indi.solution for indi in population.pop]
	with open(f'MOEAD_Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/dataset_{dataset_no}/first_solutions_{epoch}.pickle','wb') as file:
		pickle.dump(first_solutions,file)

	for i in tqdm(range(NUM_GENERATION)):
		population.reproduct()
		f = []
		fitnesses = []
		best_fitness = 0
		for indi in population.pop:
			f.append(copy.deepcopy(indi.f))
			
		objectives_by_generations.append(f)

	
	last_solutions = [indi.solution for indi in population.pop]
	
	# Change file name everytime!
	with open(f'MOEAD_Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/dataset_{dataset_no}/last_solutions_{epoch}.pickle','wb') as file:
		pickle.dump(last_solutions,file)
	with open(f'MOEAD_Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/dataset_{dataset_no}/objectives_by_generations_{epoch}.pickle','wb') as file:
		pickle.dump(objectives_by_generations,file)
	with open(f'MOEAD_Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/dataset_{dataset_no}/lambdas_{epoch}.pickle','wb') as file:
		pickle.dump(population.lambdas,file)