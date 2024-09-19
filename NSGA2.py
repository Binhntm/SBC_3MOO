import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
import copy
import time

# 1 Individual contains 1 sub-problem and 1 solution
class Individual:
    def __init__(self, num_sensors, num_sink_nodes, sensors_positions, sink_node_positions, distances, barrier_length=1000, solution = None, active_indx=None, f=None) -> None:
        self.num_sensors = num_sensors
        self.num_sink_nodes = num_sink_nodes
        self.sensors_positions = sensors_positions
        self.sink_nodes_positions = sink_node_positions
        self.distances = distances
        self.barrier_length = barrier_length

        self.active_indx = []   # Index of active sensor
        
        self.solution = solution
        if self.solution is None:
            # Random solution
            self.solution = np.zeros((self.num_sensors,2))
            activate = np.random.choice([0,1],num_sensors)
            srange = np.random.rand(num_sensors)*barrier_length/num_sensors
            for i in range(num_sensors):
                self.solution[i,0] = activate[i]
                self.solution[i,1] = activate[i]*srange[i]
            self.repair_solution()
            self.compute_objectives(self.solution)

        else:   # Initialize from an existed instance
            self.solution = solution
            self.active_indx = active_indx
            self.f = f


    @staticmethod
    def euclid_distance(point1, point2):
        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


    def compute_objectives(self, solution):
        f = np.zeros(3)

        for i in range(self.num_sensors):
            if(solution[i][0]==1):
                # self.active_indx is always computed before in the self.repair_solution function
                # self.active_indx.append(i)
                f[1] += 1

                nearest_sink_node_distance = 1e9
                for j in range(self.num_sink_nodes):
                    distance = self.euclid_distance(self.sensors_positions[i], self.sink_nodes_positions[j])
                    nearest_sink_node_distance = min(nearest_sink_node_distance, distance)
            
                f[2] += nearest_sink_node_distance
              
        f[2]/=f[1]
        
        # d_{0,1} and d{k,k+1}
        dfirst = self.sensors_positions[self.active_indx[0]][0]*0.5
        dlast = (1000 - self.sensors_positions[self.active_indx[-1]][0])*0.5
        if len(self.active_indx)==1:
            f[0] = dfirst**2 + dlast**2 + self.solution[self.active_indx[0]][1]**2
        else:
            for i in range(len(self.active_indx)):
                indx, left_adj_indx, right_adj_indx = i, i-1, i+1
                if i==0:
                    f[0] += dfirst**2*0.5 + self.solution[indx][1]**2 + self.euclid_distance(self.sensors_positions[indx], self.sensors_positions[right_adj_indx])**2*0.5
                elif i==len(self.active_indx)-1:
                    f[0] += dlast**2*0.5 + self.solution[indx][1]**2 + self.euclid_distance(self.sensors_positions[indx], self.sensors_positions[left_adj_indx])**2*0.5
                else:
                    f[0] += self.solution[indx][1]**2 + 0.5*(self.euclid_distance(self.sensors_positions[indx], self.sensors_positions[left_adj_indx])**2 + self.euclid_distance(self.sensors_positions[indx], self.sensors_positions[right_adj_indx])**2)

        self.f = f.copy()

    
    def dominate(self, competition_obj:np.ndarray):
        '''
        Check if this Individual dominates another Individual
        '''
        smaller_or_equal = self.f <= competition_obj
        smaller = self.f < competition_obj
        if np.all(smaller_or_equal) and np.any(smaller):
            return True

        return False
    
    def mutation(self):
        active_sensor_index = []
        sleep_sensor_index = []
        for i in range(len(self.solution)):
            if(self.solution[i][0]==1):
                active_sensor_index.append(i)
            else:
                sleep_sensor_index.append(i)

        # Choose a sleep sensor and a active sensor, then exchange their state
        change_index = np.random.choice(active_sensor_index), np.random.choice(sleep_sensor_index)

        temp = self.solution[change_index[0]]
        self.repair_solution()
        self.solution[change_index[0]] = self.solution[change_index[1]]
        self.solution[change_index[1]] = temp
        

        return
                   
    def repair_solution(self):
        # Get index of active sensors
        self.active_indx = []
        # Distance between active adjacent sensors 
        def find_active_indx():
            self.active_indx = []
            for i in range(len(self.sensors_positions)):            
                if(self.solution[i][0]==1):
                    self.active_indx.append(i)

        find_active_indx()
        # Coverage requirement
        if len(self.active_indx)==0:
            random_active = np.random.randint(0,self.num_sensors)
            self.solution[random_active][0] = 1
            self.active_indx.append(random_active)

        if len(self.active_indx)==1:
            self.solution[self.active_indx[0]][1] = max(
                self.sensors_positions[self.active_indx[0]][0],   # x (is tangent one edge of ROI)
                self.barrier_length - self.sensors_positions[self.active_indx[0]][0]  # barrier_length - x
            )
            return

        # Coverage in two ends of ROI
        self.solution[self.active_indx[0]][1] = max(
            self.sensors_positions[self.active_indx[0]][0], # x (is tangent one edge of ROI)
            self.distances[self.active_indx[0], self.active_indx[1]]/2
        )
        self.solution[self.active_indx[-1]][1] = max(
            self.barrier_length - self.sensors_positions[self.active_indx[-1]][0], # barrier_length - x (is tangent one edge of ROI)
            self.distances[self.active_indx[-1], self.active_indx[-2]]/2
        )

        for i in range(1,len(self.active_indx)-1):
            self.solution[self.active_indx[i]][1] = max(
                self.distances[self.active_indx[i], self.active_indx[i-1]]/2,
                self.distances[self.active_indx[i+1], self.active_indx[i]]/2
            )

        # Shrink
        ## Prune sensors in between
        length = len(self.active_indx)
        i = 1
        while(i<length-1):  # O(n)
            if(
                self.distances[self.active_indx[i],self.active_indx[i-1]] + self.solution[self.active_indx[i]][1] <= self.solution[self.active_indx[i-1]][1]   # sensor {active_index[i]} covered by left adjacent
                or
                self.distances[self.active_indx[i],self.active_indx[i+1]] + self.solution[self.active_indx[i]][1] <= self.solution[self.active_indx[i+1]][1]   # sensor {active_index[i]} covered by right adjacent
                or 
                self.distances[self.active_indx[i-1],self.active_indx[i+1]] <= self.solution[self.active_indx[i-1]][1] + self.solution[self.active_indx[i+1]][1]   # two of its adjacent intersect
            ):
                self.solution[self.active_indx[i]] = [0,0]
                self.active_indx.pop(i)
                length-=1
                if(i>1):
                    i-=1
                continue
            i+=1
        
        ## Prune sensors at two ends
        if length>1:
            if self.solution[self.active_indx[1]][1] >= self.sensors_positions[self.active_indx[1]][0]: # right adjacent of most-left sensor reached the left side of ROI 
                    self.solution[self.active_indx[0]] = [0,0]
                    self.active_indx.pop(0)
                    length-=1
        if length>1:
            if self.solution[self.active_indx[-2]][1] + self.sensors_positions[self.active_indx[-2]][0] >= self.barrier_length:
                    self.solution[self.active_indx[-1]] = [0,0]
                    self.active_indx.pop(-1)
                    length-=1

        ## Shrink
        for i in range(1,len(self.active_indx)-1):
            # If sensor i's range intersect with two of its adjacents
            if(
                self.distances[self.active_indx[i], self.active_indx[i-1]] < self.solution[self.active_indx[i]][1] + self.solution[self.active_indx[i-1]][1]
                and
                self.distances[self.active_indx[i], self.active_indx[i+1]] < self.solution[self.active_indx[i]][1] + self.solution[self.active_indx[i+1]][1]
            ):
                # The distance between sensor i and i-1's range: d1 = distance(sensor_i, sensor_i-1) - R(sensor_i-1)
                d1 = self.distances[self.active_indx[i],self.active_indx[i-1]] - self.solution[self.active_indx[i-1]][1]
                # The distance between sensor i and i+1's range: d2 = distance(sensor_i, sensor_i+1) - R(sensor_i+1)
                d2 = self.distances[self.active_indx[i],self.active_indx[i+1]] - self.solution[self.active_indx[i+1]][1]

                self.solution[self.active_indx[i]][1] = max(d1,d2)

        if length>1:
            if self.distances[self.active_indx[0], self.active_indx[1]] < self.solution[self.active_indx[0]][1] + self.solution[self.active_indx[1]][1]:
                self.solution[self.active_indx[0]][1] = max(
                    self.sensors_positions[self.active_indx[0]][0], # Ensure it reach border of ROI
                    self.distances[self.active_indx[0], self.active_indx[1]] - self.solution[self.active_indx[1]][1])
        if length>1:
            if self.distances[self.active_indx[-1], self.active_indx[-2]] < self.solution[self.active_indx[-1]][1] + self.solution[self.active_indx[-2]][1]:
                self.solution[self.active_indx[-1]][1] = max(
                    self.barrier_length - self.sensors_positions[self.active_indx[-1]][0],
                    self.distances[self.active_indx[-1], self.active_indx[-2]] - self.solution[self.active_indx[-2]][1])
            
        return


class Population:
    def __init__(self, pop_size, num_sensors, sensors_positions,num_sink_nodes, sink_nodes_positions, barrier_length=1000, mu=1, coverage_threshold=0, crossoverate=0.9, mutation_rate=0.4) -> None:
        self.pop_size = pop_size
        self.num_sensors = num_sensors
        self.sensors_positions = sensors_positions
        self.num_sink_nodes = num_sink_nodes
        self.sink_nodes_positions = sink_nodes_positions
        self.pop:list[Individual] = []
        self.EP = []
        self.distances = np.zeros(shape=(self.num_sensors, self.num_sensors))
        self.coverage_threshold = coverage_threshold
        self.barrier_length = barrier_length
        self.crossoverate = crossoverate
        self.mutation_rate = mutation_rate

        for i in range(num_sensors):
            for j in range(num_sensors):
                d =  Individual.euclid_distance(self.sensors_positions[i], self.sensors_positions[j])

                self.distances[i,j] = self.distances[j,i] = d

        for i in range(self.pop_size):
            indi = Individual(num_sensors, self.num_sink_nodes, sensors_positions, sink_nodes_positions, self.distances)
            indi = Individual(num_sensors, num_sink_nodes, sensors_positions, sink_nodes_positions, self.distances, barrier_length)
            self.pop.append(indi)

    def new_individual(self, individual:Individual)->Individual:
        # Pass by value
        sol = individual.solution.copy()
        act_indx = copy.deepcopy(individual.active_indx)
        f = individual.f.copy()
        new = Individual(self.num_sensors, self.num_sink_nodes, self.sensors_positions, self.sink_nodes_positions, self.distances, self.barrier_length, sol, act_indx, f)

        return new

    def __repr__(self) -> str:
        # print every individual in population
        res = ""
        for i in range(self.pop_size):
            res += f"Solution to Individual {i}: {self.pop[i].solution}\n"
        return res
    
  
    def forward_local_search(self, individual:Individual):
        sorted_gene_index = [i for i, gene in sorted(enumerate(individual.solution), key=lambda gene: gene[1],reverse=True)]
        new_sol = self.new_individual(individual)
        for index in sorted_gene_index:
            if(index!=0 and index!=len(individual.solution)-1 
               and individual.solution[index][0]==1
               and individual.solution[index-1][0]==0
               and individual.solution[index+1][0]==0):

                d1 = np.sqrt((individual.sensors_positions[index][0]-individual.sensors_positions[index-1][0])**2 + (individual.sensors_positions[index][1]-individual.sensors_positions[index-1][1])**2) 

                d2 = np.sqrt((individual.sensors_positions[index][0]-individual.sensors_positions[index+1][0])**2 + (individual.sensors_positions[index][1]-individual.sensors_positions[index+1][1])**2) 

                r1 = individual.solution[0][1] - d1
                r2 = individual.solution[0][1] - d2

                new_sol.solution[index] = [0,0]
                new_sol.solution[index-1][0] = 1
                new_sol.solution[index+1][0] = 1

                new_sol.solution[index-1][1] = r1
                new_sol.solution[index+1][1] = r2

                break
        
        new_sol.repair_solution()
        new_sol.compute_objectives(new_sol.solution)
        
        return new_sol

    def backward_local_search(self, individual:Individual):
        active_index = []
        for i in range(len(individual.solution)):
            if(individual.solution[i][0]==1):
                active_index.append(i)
        if(len(active_index)<3):
            return
        d_min = np.inf
        turn_off = [] # Containts index of 2 sensors that will be deactivate
        turn_on = 0 # Containst index of sensor that will be activate
        for i in range(len(active_index)-1):
            # Get 2 sensors have smallest sum range 
            if(individual.solution[active_index[i]][1] + individual.solution[active_index[i+1]][1]<d_min):
                # Check if that 2 sensors have another sleep sensors in between
                if(active_index[i+1] - active_index[i] > 1):
                    d_min = individual.solution[active_index[i+1]][1] + individual.solution[active_index[i]][1]
                    turn_off.clear()
                    turn_off.append(active_index[i])
                    turn_off.append(active_index[i+1])
                    turn_on = int((active_index[i]+active_index[i+1])/2)

        new_sol = self.new_individual(individual)
        d1 = np.sqrt((individual.sensors_positions[turn_on][0]-individual.sensors_positions[turn_off[0]][0])**2 + (individual.sensors_positions[turn_on][1]-individual.sensors_positions[turn_off[0]][1])**2) 

        d2 = np.sqrt((individual.sensors_positions[turn_on][0]-individual.sensors_positions[turn_off[1]][0])**2 + (individual.sensors_positions[turn_on][1]-individual.sensors_positions[turn_off[0]][1])**2) 
        
        r1 = d1 + individual.solution[turn_off[0]][1]
        r2 = d2 + individual.solution[turn_off[1]][1]
        r = max(r1,r2)

        new_sol.solution[turn_off[0]] = [0,0]
        new_sol.solution[turn_off[1]] = [0,0]
        new_sol.solution[turn_on] = [1,r]

        new_sol.repair_solution()
        new_sol.compute_objectives(new_sol.solution)

        return new_sol

    def local_search(self, k):
        if(k<self.pop_size-1):
            # idea: sub-problem k+1 assigns smaller weight to f2 
            self.forward_local_search(self.pop[k])
        if(k>0):
            self.backward_local_search(self.pop[k])   


    def selection(self,num_indi=10, k=16):
        '''
        Return an Individual and its numerical order in population
        '''
        # k is number of individuals in selection pool
        indi_index = list(np.random.choice(range(0,self.pop_size),size=k))
        pool = [[self.pop[i], 1, i] for i in indi_index]        
        # sort pool by sub-problem's utility, take last element
        return np.array(sorted(pool, key=lambda x: x[1])[-1:-1-num_indi]).transpose()
    
    # copy first half of solution from individual, second half from breed to new_individual
    def crossover(self, individual:Individual, breed:Individual)->Individual:
        # random 1 point crossover
        cross_point = random.randint(0,len(individual.solution)-1)

        new_individual = self.new_individual(individual)
        all_off = True
        while (all_off==True):
            for i in range(self.num_sensors):
                if(i>=cross_point):
                    new_individual.solution[i] = copy.deepcopy(breed.solution[i])
                if(new_individual.solution[i][0]==1):
                    all_off = False

        new_individual.compute_objectives(new_individual.solution)
        return new_individual
    
    def uniform_crossover(self, parent1:Individual, parent2:Individual)->list[Individual,Individual]:
        rand = np.random.uniform(0,1,self.num_sensors)

        child1 , child2 = self.new_individual(parent1), self.new_individual(parent2)
        child1.solution = np.where(rand[:, np.newaxis] >= 0.5, parent2.solution, child1.solution)
        child2.solution = np.where(rand[:, np.newaxis] >= 0.5, parent1.solution, child2.solution)

        # child1.compute_objectives(child1.solution)
        # child2.compute_objectives(child2.solution)
        child1.repair_solution()

        return child1, child2
    
    def two_point_crossover(self, parent1:Individual, parent2:Individual)->list[Individual,Individual]:
        cross_point = np.random.randint(1,self.num_sensors-1,2)
        cross_point.sort()
        child1 , child2 = self.new_individual(parent1), self.new_individual(parent2)
        all_off = [True,True]
        for i in range(0,self.num_sensors):
            if(i>=cross_point[0] and i<=cross_point[1]):
                child1.solution[i] = copy.deepcopy(parent2.solution[i])
                child2.solution[i] = copy.deepcopy(parent1.solution[i])
            if(child1.solution[i][0]==1):
                all_off[0] = False
            if(child2.solution[i][0]==1):
                all_off[1] = False

        if(any(all_off)):
            child1, child2 = self.two_point_crossover(parent1,parent2)
        child1.compute_objectives(child1.solution)
        child2.compute_objectives(child2.solution)

        return child1, child2
    
    def update_EP(self, individual: Individual):
        new_EP = []
        add_to_EP = True

        if len(self.EP) == 0:
            self.EP.append(individual)
            return

        # loop through EP to find solutions dominated by individual, and find solutions that dominate individual
        # if individual dominates a solution in EP, replace that solution with individual
        # if individual is dominated by a solution in EP, do nothing
        for solution in self.EP:
            dominated_by_individual = False
            dominate_individual = False
            for j in range(3):
                if solution.f[j] > individual.f[j]:
                    dominated_by_individual = True
                    break
                elif solution.f[j] < individual.f[j]:
                    dominate_individual = True
            if not dominated_by_individual:
                new_EP.append(solution)
                if dominate_individual:
                    add_to_EP = False

        if add_to_EP:
            new_EP.append(individual)

        self.EP = new_EP
        

    
    def reproduct(self):
        new_children:list[Individual] = []
        new_children_size = self.pop_size//4

        # Select sub-problem  
        pool_size = self.pop_size
        rand = np.random.permutation(self.pop_size)[:pool_size]
        random_crossover = np.random.uniform(0,1, new_children_size)
        random_mutation = np.random.uniform(0,1, new_children_size)
        for i in range(int(new_children_size)):
            # # Offspring generation 
            if random_crossover[i] < self.crossoverate:
                child, _ = self.uniform_crossover(self.pop[rand[i]],self.pop[rand[-i]])
            else:
                continue

            if random_mutation[i] < self.mutation_rate:
                child.mutation()

            child.compute_objectives(child.solution)
            new_children.append(child)
        # Assume the id of individual in pool is followed by self.pop append new_children 
        pool:list[Individual] = self.pop + new_children
        
        # Ranking
            # domination_sets[i] is a list, containts the ids of which individual i dominates
        domination_sets = [[] for i in range(len(pool))]
            # domination_counts[i] a number, containts the numbers of individual dominate i
        domination_counts = np.zeros(len(pool))
        for i in range(len(pool)):
            for j in range(i+1,len(pool)):
                if(pool[i].dominate(pool[j].f)):
                    domination_sets[i].append(j)
                    domination_counts[j]+=1
                elif(pool[j].dominate(pool[i].f)):
                    domination_sets[j].append(i)
                    domination_counts[i]+=1

        # Contains index of solution in pool, eg: pareto_front = [[0,2,3],[4,1]] means that individuals pool[0], pool[2], pool[3] are in rank 0, pool[4] and pool[1] ar in rank 1
        pareto_front:list[list[int]] = []  
        while True:
            current_front = np.where(domination_counts==0)[0]
            if(len(current_front)==0):
                break
            pareto_front.append(current_front)

            for i in current_front:
                domination_counts[i] = -1

                for j in domination_sets[i]:
                    domination_counts[j] -= 1

        new_pop = [self.new_individual(self.pop[i]) for i in range(self.pop_size)]

        count = 0
        for rank in pareto_front:
            if(count+len(rank)<self.pop_size):
                for indi_index in rank:
                    # new_pop[count].solution = [copy.deepcopy(row) for row in pool[indi_index].solution]
                    # new_pop[count].f = copy.deepcopy(pool[indi_index].f)
                    new_pop[count].solution = pool[indi_index].solution.copy()
                    new_pop[count].f = pool[indi_index].f.copy()

                    count += 1
            
            else:
                # Calculate crowding distances 
                distances = np.zeros(len(pool))
                for obj_index in range(3):
                    objectives = [[indi_index, pool[indi_index].f[obj_index]] for indi_index in rank]
                    objectives = sorted(objectives,key=lambda x:x[1])

                    distances[objectives[0][0]] = np.inf
                    distances[objectives[-1][0]] = np.inf

                    f_min = objectives[0][1]
                    f_max = objectives[-1][1]

                    if(f_max==f_min):
                        continue

                    for i in range(1,len(objectives)-1):
                        # The indexes and object function of 2 adjacent solution
                        prev_index, prev_obj = objectives[i-1]
                        next_index, next_obj = objectives[i+1]
                        index, obj = objectives[i]

                        distances[index] += (next_obj - prev_obj)/(f_max-f_min)
                
                sorted_index = np.flip(np.argsort(distances))

                for i in range(self.pop_size-count):
                    # new_pop[i+count].solution = [copy.deepcopy(row) for row in pool[sorted_index[i]].solution]
                    # new_pop[i+count].f = copy.deepcopy(pool[sorted_index[i]].f)
                    new_pop[i+count].solution = pool[sorted_index[i]].solution.copy()
                    new_pop[i+count].f = pool[sorted_index[i]].f.copy()
                break

        self.pop = new_pop
        return
