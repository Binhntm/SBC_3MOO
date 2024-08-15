import numpy as np
import pandas as pd
import pickle
import pygmo as pg
from sklearn.preprocessing import MinMaxScaler

def open(link):
    try:
        with open(link,'rb') as file:
            obj = np.array(pickle.load(file))

            return obj
    except:
        print('Wrong link or file not exist!')

def find_pareto_front(pool):
    def dominate(obj1, obj2):
        '''
        Check if this obj1 dominates obj2
        '''
        obj1 = np.array(obj1)
        obj2 = np.array(obj2)
        smaller_or_equal = obj1 <= obj2
        smaller = obj1 < obj2
        if np.all(smaller_or_equal) and np.any(smaller):
            return True

        return False

    # Ranking
        # domination_sets[i] is a list, containts the ids of which individual i dominates
    domination_sets = [[] for i in range(len(pool))]
        # domination_counts[i] a number, containts the numbers of individual dominate i
    domination_counts = np.zeros(len(pool))
    for i in range(len(pool)):
        for j in range(i+1,len(pool)):
            if(dominate(pool[i],pool[j])):
                domination_sets[i].append(j)
                domination_counts[j]+=1
            elif(dominate(pool[j],pool[i])):
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

    # Return rank 0
    res = []
    for indi_index in pareto_front[0]:
        res.append(pool[indi_index])

    return res

def scale_objectives(objectives, nadir_point, ideal_point):
    scaled_objectives = []
    for indi_objectives in objectives:
        scaled = [0,0,0]
        for i in range(3):
            scaled[i] = (indi_objectives[i]-ideal_point[i])/(nadir_point[i]-ideal_point[i])
        scaled_objectives.append(scaled)

    return scaled_objectives

def hypervolume(output_objectives, nadir_point):
    '''
    # Calculate the Hypervolume of a set of output objectives vectors with provided nadir point 
    The output objectives vectors and nadir point should be scaled beforehand

    Belong to the convergence and distribution category
    '''
    hyper_volume = pg.hypervolume(output_objectives).compute(nadir_point)

    return hyper_volume

def spacing_indicator(output_objecvies):
    '''
    # Calculate the Spacing indicator of a set of output objectives vectors
    The output objectives vectors should be scaled beforehand

    Belong to the distribution and spread category
    '''
    d_min = np.full(len(output_objecvies),np.Infinity)

    for i in range(len(output_objecvies)):
        for j in range(len(output_objecvies)):
            if(i!=j):
                d_min[i] = min(d_min[i], np.linalg.norm(output_objecvies[i]-output_objecvies[j]))
    
    d_avg = np.mean(d_min)
    
    sum = 0
    for i in range(len(output_objecvies)):
        sum += (d_avg-d_min[i])**2
   
    return np.sqrt(sum/(len(output_objecvies)-1))

def delta_indicator():
    '''
    # The delta index does not generalize to more than 2 objectives, as it uses lexicographic order in the biobjective objective space to compute the di

    '''
    return

def generational_distance(output_objectives, reference_pareto_front, p=2):
    '''
    # Calculate the Genration distace of a set of output objectives vectors with provied pseudo Pareto front
    The output objectives vectors should be scaled beforehand

    Belong to the convergence category
    '''

    sum = 0
    for i in range(len(output_objectives)):
        min_distance = np.Infinity
        for j in range(len(reference_pareto_front)):
            min_distance = min(np.linalg.norm(output_objectives[i]-reference_pareto_front[j]), min_distance)
        sum+=min_distance**p
        
    return 1/len(output_objectives)*sum**(1/p)

def inverted_generational_distance(output_objectives, reference_pareto_front, p=2):
    '''
    # Calculate the Inverted genration distace of a set of output objectives vectors with provied pseudo Pareto front
    The output objectives vectors should be scaled beforehand

    Belong to the convergence category
    '''
    sum = 0
    for i in range(len(reference_pareto_front)):
        min_distance = np.Infinity
        for j in range(len(output_objectives)):
            min_distance = min(np.linalg.norm(reference_pareto_front[i]-output_objectives[j]), min_distance)
        sum+=min_distance**p
        
    return 1/len(reference_pareto_front)*sum**(1/p)

def epsilon_additive_indicator(output_objectives, reference_pareto_front):
    '''
    # Calculate the Epsilon additive indicator of a set of output objectives vectors with provied pseudo Pareto front
    The output objectives vectors should be scaled beforehand

    Belong to convergence category
    Used instead of epsilon indicator because after scale, some values come close to zero
    '''
    max_val = -np.Infinity
    for x2 in output_objectives:
        min_val = np.Infinity
        for x1 in reference_pareto_front:
            try:
                min_val = min(min_val, max([x2[i]-x1[i] for i in range(3)]))

            except Exception as e: 
                print(f'Error: {e}')
                print(f'output: {x2},\nreference: {x1}')
        max_val = max(max_val, min_val)

    return max_val

