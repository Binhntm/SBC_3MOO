import numpy as np
import scipy.stats as stats
import pickle
import yaml
import os

def generate_positions(number_sensors, length=1000, width=50, distribution='uniform'):
    if(distribution=='uniform'):
        # x-coordinate from 0 to length of border
        sensors_x = np.sort(np.random.uniform(low=0,high=length,size=(number_sensors)),axis=0)
        # y-coordinate from -width/2 to width/2
        sensors_y = np.random.uniform(low=-width/2,high=width/2,size=(number_sensors))
        sensors_positions = np.array([[sensors_x[i],sensors_y[i]] for i in range(number_sensors)])

    if(distribution=='gauss'):
        mu = length/2  # Mean of the Gaussian distribution, set to half of the length
        sigma = mu/4  # Standard deviation of the Gaussian distribution, set to one-forth of the mean
        lower, upper = 0, length
        # Generating x-coordinates using a truncated normal distribution
        sensors_x = np.sort(np.array(stats.truncnorm((lower-mu)/sigma,(upper-mu)/sigma,loc=mu, scale=sigma).rvs(number_sensors)),axis=0)
        sensors_y = np.random.uniform(low=-width/2,high=width/2,size=(number_sensors))
        sensors_positions = np.array([[sensors_x[i],sensors_y[i]] for i in range(number_sensors)])

    if(distribution=='exponential'):
        # Lambda for exponential distribution controls the rate parameter (inverse of mean)
        scale = length / number_sensors  # Adjust scale based on number of sensors and length
        sensors_x = np.sort(np.random.exponential(scale=scale, size=(number_sensors)))
        # Clip values to stay within 0 and the length boundary
        sensors_y = np.random.uniform(low=-width/2, high=width/2, size=(number_sensors))
        sensors_positions = np.array([[sensors_x[i], sensors_y[i]] for i in range(number_sensors)])

    return sensors_positions

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    length = config['DIMENSIONS']['LENGTH']
    width = config['DIMENSIONS']['WIDTH']
    
    for distr in ['uniform', 'gauss', 'exponential']:
        for i in range(5):
            for num_sensors in [100,300,700]:
                sensors_positions = generate_positions(num_sensors, length, width, distr)
                with open(f'Datasets/{distr}/{width}x{length}unit/{num_sensors}sensors/sensors_positions_{i}.pickle','wb') as file:
                    pickle.dump(sensors_positions,file)


    with open('Datasets/sink_nodes_positions.pickle','wb') as file:
            sink_node_pos = [[500,-100]]
            pickle.dump(sink_node_pos,file)
        