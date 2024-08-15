import MOEAD
import numpy as np
import matplotlib.pyplot as plt
import Plot
import pickle
import copy

if __name__ == "__main__":
	POP_SIZE = 20
	NEIGHBORHOOD_SIZE = 3
	NUM_SENSORS = 100
	NUM_SINK_NODES = 1
	NUM_GENERATION = 30000
	LENGTH, WIDTH = 1000, 50
	NUM_EPOCH = 5

	# Load positions
	dataset_no = 0
	with open(f'Datasets/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/sensors_positions_{dataset_no}.pickle','rb') as file:
		sensors_positions = pickle.load(file)
	with open('Datasets/sink_nodes_positions.pickle','rb') as file:
		sink_nodes_positions = pickle.load(file)
	
	# Run
	for epoch in range(NUM_EPOCH):
		population = MOEAD.Population(POP_SIZE,NEIGHBORHOOD_SIZE,NUM_SENSORS,sensors_positions,NUM_SINK_NODES,sink_nodes_positions)

		best_indi_fitness = []
		pop_avg_fitness = []
		objectives_by_generations = []
		first_solutions = [indi.solution for indi in population.pop]
		with open(f'Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/dataset_{dataset_no}/first_solutions_{epoch}.pickle','wb') as file:
			pickle.dump(first_solutions,file)

		for i in range(NUM_GENERATION):
			pop_fitness = [indi.fitness for indi in population.pop]	
			population.reproduct()
			f = []
			for indi in population.pop:
				f.append(copy.deepcopy(indi.f_norm))
			objectives_by_generations.append(f)
			best = sorted(population.pop,key= lambda x:x.fitness)[-1]
			best_indi_fitness.append(best.fitness)
			pop_avg_fitness.append(np.mean(pop_fitness))

			if(i%100==0):
				print(f'epoch {epoch}: {i/NUM_GENERATION*100}%')
		
		last_solutions = [indi.solution for indi in population.pop]
		
		# Change file name everytime!
		with open(f'Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/dataset_{dataset_no}/last_solutions_{epoch}.pickle','wb') as file:
			pickle.dump(last_solutions,file)
		with open(f'Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/dataset_{dataset_no}/objectives_by_generations_{epoch}.pickle','wb') as file:
			pickle.dump(objectives_by_generations,file)
		with open(f'Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/dataset_{dataset_no}/pop_avg_fitness_{epoch}.pickle','wb') as file:
			pickle.dump(pop_avg_fitness,file)
		with open(f'Results/uniform/{WIDTH}x{LENGTH}unit/{NUM_SENSORS}sensors/dataset_{dataset_no}/best_indi_fitness_{epoch}.pickle','wb') as file:
			pickle.dump(best_indi_fitness,file)
		