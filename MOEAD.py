import numpy as np
from sklearn.neighbors import NearestNeighbors
import copy

# 1 Individual contains 1 sub-problem and 1 solution
class Individual:
    def __init__(self,lambdas, num_sensors, num_sink_nodes, sensors_positions, sink_node_positions, distances, mu=1, coverage_threshold=0, barrier_length=1000, solution=None, active_indx=None, fitness=None) -> None:
        self.num_sensors = num_sensors
        self.num_sink_nodes = num_sink_nodes
        self.lambdas = lambdas
        self.sensors_positions = sensors_positions
        self.sink_nodes_positions = sink_node_positions
        self.mu = mu
        self.distances = distances
        self.barrier_length = barrier_length
        self.coverage_threshold = coverage_threshold
        
        self.active_indx = []   # Index of active sensor

        self.f_norm = np.array([1e9,1e9,1e9])
        self.f = np.array([1e9,1e9,1e9])

        self.neighbor:list[Individual] = []  # To-do: Review

        self.distances = distances
        if solution is None:   
            # Random solution
            self.solution = np.zeros((self.num_sensors,2))
            activate = np.random.choice([0,1],num_sensors)
            srange = np.random.rand(num_sensors)*barrier_length/num_sensors
            for i in range(num_sensors):
                self.solution[i,0] = activate[i]
                self.solution[i,1] = activate[i]*srange[i]
            self.repair_solution()
            self.fitness = self.compute_fitness(self.solution, None, None)

        else:   # Initialize from an existed instance
            self.solution = solution
            self.active_indx = active_indx
            self.fitness = fitness

    @staticmethod
    def euclid_distance(point1, point2):
        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


    def compute_fitness(self, solution, ideal_point, nadir_point):
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
        
        
        # Normalizing
        self.f = f.copy()
        if ideal_point is None: # Skip normalizing in the first loop
            self.f_norm = f.copy()
            gte = max(self.lambdas*self.f_norm)
        else:
            f = (f-np.array(ideal_point))/(np.array(nadir_point)-np.array(ideal_point))
            # f = (f-np.array(ideal_point))
            self.f_norm = f.copy()
            # gte = max(self.lambdas*abs(f-ideal_point))
            gte = max(self.lambdas*self.f_norm)

        self.fitness = -gte
        return self.fitness
    
    def mutation(self):
        # To-do: Optimize performance
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
    

    def update_utility(self, new_fitness):
        prev_fitness = self.fitness
        delta_i = new_fitness - prev_fitness

        if(delta_i>0.001):
            self.mu = 1
        else:
            self.mu = 0.99 + 0.01*delta_i / 0.001
        # print("update utility", self.mu, delta_i)

    def add_neighbor(self, individual):
        self.neighbor.append(individual)

class Population:
    def __init__(self, pop_size, neighborhood_size, num_sensors, sensors_positions, num_sink_nodes, sink_nodes_positions, barrier_length=1000, mu=1, coverage_threshold=0, crossover_rate=0.9, mutation_rate=0.4) -> None:
        self.pop_size = pop_size
        self.neighborhood_size = neighborhood_size
        self.num_sensors = num_sensors
        self.sensors_positions = sensors_positions
        self.num_sink_nodes = num_sink_nodes
        self.sink_nodes_positions = sink_nodes_positions
        self.lambdas = self.generate_lambdas()
        self.pop:list[Individual] = []
        self.ideal_point = [np.inf,np.inf,np.inf]
        self.nadir_point = [0,0,0]
        # self.ideal_point = [0,0,0]
        # self.nadir_point = [1e10,num_sensors,]
        self.EP:list[Individual] = []
        self.distances = np.zeros(shape=(self.num_sensors, self.num_sensors))   # Distance between two sensors
        self.barrier_length= barrier_length
        self.mu = mu
        self.coverage_threshold = coverage_threshold
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        for i in range(num_sensors):
            for j in range(num_sensors):
                d = Individual.euclid_distance(self.sensors_positions[i], self.sensors_positions[j])

                self.distances[i,j] = self.distances[j,i] = d

        for i in range(self.pop_size):
            indi = Individual(self.lambdas[i], num_sensors, self.num_sink_nodes, sensors_positions, sink_nodes_positions, self.distances, self.mu, self.coverage_threshold, self.barrier_length)
            self.pop.append(indi)

        def find_neighbor():
            # max value for distance to neighbor
            X = np.array(self.lambdas)
            nbrs = NearestNeighbors(n_neighbors=self.neighborhood_size, algorithm='ball_tree').fit(X)
            distances, indices = nbrs.kneighbors(X)
            for i in range(len(self.lambdas)):
                for j in indices[i]:
                    self.pop[i].add_neighbor(self.pop[j])
        
        find_neighbor()
        self.update_ideal_point()

    def new_individual(self, individual:Individual)->Individual:
        # Pass by value
        sol = individual.solution.copy()
        act_indx = copy.deepcopy(individual.active_indx)
        new = Individual(individual.lambdas, individual.num_sensors, individual.num_sink_nodes, individual.sensors_positions, individual.sink_nodes_positions, self.distances, self.mu, self.coverage_threshold, self.barrier_length, sol, act_indx, individual.fitness)

        return new

    def __repr__(self) -> str:
        # print every individual in population
        res = ""
        for i in range(self.pop_size):
            res += f"Solution to Individual {i}: {self.pop[i].solution}\n"
        return res
    
    # Genrate uniformly spread weighted vectors lambda 
    def generate_lambdas(self):
        weights = []
        lambdas = np.random.uniform(0,1,(self.pop_size,3))
        for i in range(self.pop_size):
            weights.append([lambdas[i,j]/sum(lambdas[i]) for j in range(3)])

        return weights
    
  
    def forward_local_search(self, individual:Individual):
        '''
        # Find the sensor that has the largest range, turn it off and turn on the two of its adjacent
        '''
        sorted_gene_index = individual.solution[:,1].argsort()[::-1]
        
        new_sol = self.new_individual(individual)
        for index in sorted_gene_index:
            if(index!=0 and index!=len(individual.solution)-1 
               and individual.solution[index][0]==1
               and individual.solution[index-1][0]==0 and individual.solution[index+1][0]==0):

                d1 = np.sqrt((individual.sensors_positions[index][0]-individual.sensors_positions[index-1][0])**2 + (individual.sensors_positions[index][1]-individual.sensors_positions[index-1][1])**2) 

                d2 = np.sqrt((individual.sensors_positions[index][0]-individual.sensors_positions[index+1][0])**2 + (individual.sensors_positions[index][1]-individual.sensors_positions[index+1][1])**2) 

                r1 = individual.solution[0][1] - d1
                r2 = individual.solution[0][1] - d2

                new_sol.solution[index] = [0,0]
                new_sol.solution[index-1] = [1, r1]
                new_sol.solution[index+1] = [1, r2]

                break
        
        new_sol.compute_fitness(new_sol.solution, self.ideal_point, self.nadir_point)
        if(new_sol.fitness>individual.fitness):
            individual.solution = new_sol.solution.copy()
            individual.fitness = new_sol.fitness
            individual.f_norm = new_sol.f_norm.copy()
            individual.f = new_sol.f.copy()

        return 

    def backward_local_search(self, individual:Individual):
        if(len(individual.active_indx)<3):
            return
        d_min = np.inf
        turn_off = [] # Containts index of 2 sensors that will be deactivate
        turn_on = 0 # Containst index of sensor that will be activate
        for i in range(len(individual.active_indx)-1):
            # Get 2 sensors have smallest sum range 
            if(individual.solution[individual.active_indx[i]][1] + individual.solution[individual.active_indx[i+1]][1]<d_min):
                # Check if that 2 sensors have another sleep sensors in between
                if(individual.active_indx[i+1] - individual.active_indx[i] > 1):
                    d_min = individual.solution[individual.active_indx[i+1]][1] + individual.solution[individual.active_indx[i]][1]
                    turn_off.clear()
                    turn_off.append(individual.active_indx[i])
                    turn_off.append(individual.active_indx[i+1])
                    turn_on = int((individual.active_indx[i]+individual.active_indx[i+1])/2)

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
        new_sol.compute_fitness(new_sol.solution,self.ideal_point,self.nadir_point)
        if(new_sol.fitness>individual.fitness):
            individual.solution = new_sol.solution.copy()
            individual.fitness = new_sol.fitness
            individual.f_norm = new_sol.f_norm.copy()
            individual.f = new_sol.f.copy()

        return 

    def local_search(self, k):
        if(k<self.pop_size-1):
            # idea: sub-problem k+1 assigns smaller weight to f2 
            self.forward_local_search(self.pop[k])
        if(k>0):
            self.backward_local_search(self.pop[k])


    def selection(self, k=16)->list[Individual,int]:
        '''
        Return an Individual and its numerical order in population
        '''
        # k is number of individuals in selection pool
        indi_index = list(np.random.choice(range(0,self.pop_size),size=k))
        pool = [[self.pop[i], self.pop[i].mu, i] for i in indi_index]        
        # sort pool by sub-problem's utility, take last element
        return sorted(pool, key=lambda x: x[1])[-1]
    
    def crossover(self, parent1:Individual, parent2:Individual)->Individual:
        rand = np.random.uniform(0,1,self.num_sensors)

        child = self.new_individual(parent1)
        child.solution = np.where(rand[:, np.newaxis] >= 0.5, parent2.solution, child.solution)

        child.repair_solution()
        return child

    def update_utility(self, individuals:list[Individual]):
        for indi in individuals:
            indi.update_utility()
        return
    
    def update_neighbor_solution(self, individual:Individual):
        # get neighbor of k
        neighbors = individual.neighbor
        # evaluate solution k in neighbor sub-problems
        for neighbor in neighbors:
            new_sol = self.new_individual(neighbor)
            new_fitness = new_sol.compute_fitness(individual.solution, self.ideal_point, self.nadir_point)
            if(new_fitness>neighbor.fitness):
                neighbor.update_utility(new_fitness)
                neighbor.solution = individual.solution.copy()
                neighbor.f_norm = individual.f_norm.copy()
                neighbor.f = individual.f.copy()
                neighbor.fitness = new_fitness
    
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
        
        # Update ideal_point and z_nadir
        for i in range(3):
            self.ideal_point[i] = min(self.ideal_point[i], individual.f_norm[i])
            self.nadir_point[i] = max(self.nadir_point[i], individual.f_norm[i])

    def update_ideal_point(self):
        for individual in self.pop:
            for i in range(3):
                self.ideal_point[i] = min(self.ideal_point[i], individual.f[i])
                self.nadir_point[i] = max(self.nadir_point[i], individual.f[i])

    def reproduct(self):
        random_mutation = np.random.uniform(0,1, self.pop_size)
        random_crossover = np.random.uniform(0,1, self.pop_size)
        for i in range(self.pop_size):
            # Select pool_size sub-problem
            sub_problem, sub_problem_index = self.pop[i], i

            # Offspring generation 
            choosen_neighbor = np.random.choice(sub_problem.neighbor)
            # old_val = copy.deepcopy(sub_problem.solution)
            if random_crossover[i] < self.crossover_rate:
                child = self.crossover(sub_problem, choosen_neighbor)
            else:
                continue
            
            # Mutation
            if random_mutation[i] < self.mutation_rate:
                child.mutation()
            # Repair solution
            child.repair_solution()
            child.compute_fitness(child.solution, self.ideal_point, self.nadir_point)

            if(child.fitness >= sub_problem.fitness):
                sub_problem.update_utility(child.fitness)
                sub_problem.solution = child.solution.copy()
                sub_problem.f_norm = child.f_norm.copy()
                sub_problem.f = child.f.copy()
                sub_problem.fitness = child.fitness
                # self.update_EP(sub_problem)

            # self.local_search(sub_problem_index)
            # self.update_neighbor_solution(sub_problem)
        self.update_ideal_point()
        return
