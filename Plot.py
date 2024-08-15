import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import MOEAD
import pickle
import pyvista as pv
import plotly.express as px
import pandas as pd
import pygmo as pg

def Plot_solution(sensors_positions:list[list[float,float]], solution:list[list[int,float]]):
    '''
    # Draw a solution 
    Draw solution's all active sensors with its position and sensing range
    '''
    fix, ax = plt.subplots()
    active_x = []
    active_y = []
    for i in range(len(sensors_positions)):
        if(solution[i][0]==1):
            circle = patches.Circle((sensors_positions[i]),solution[i][1],edgecolor='blue',fill=False)
            ax.add_patch(circle)
            active_x.append(sensors_positions[i][0])
            active_y.append(sensors_positions[i][1])
    
    ax.scatter(active_x,active_y,marker='x')
    ax.set_xlim(-10,1010)
    ax.set_ylim(-100,100)
    ax.set_aspect('equal',adjustable='box')
    plt.show()

def Plot_position(sensors_positions:list[list[float,float]], sink_nodes_positions:list[list[float,float]]):
    '''
    # Draw sensors and sink nodes' positions 
    Draw sensors and sink nodes' positions
    '''
    fix, ax = plt.subplots()
    sensors_x = []
    sensors_y = []
    for i in range(len(sensors_positions)):
        sensors_x.append(sensors_positions[i][0])
        sensors_y.append(sensors_positions[i][1])
    
    sink_nodes_x = []
    sink_nodes_y = []
    for i in range(len(sink_nodes_positions)):
        sink_nodes_x.append(sink_nodes_positions[i][0])
        sink_nodes_y.append(sink_nodes_positions[i][1])

    ax.scatter(sensors_x,sensors_y,marker='x',c='red',label='Sensor')
    ax.scatter(sink_nodes_x,sink_nodes_y,marker='*',c='blue',label='Sink Node',s=20)
    ax.set_xlim(-10,1010)
    ax.set_ylim(-100,100)
    ax.set_aspect('equal',adjustable='box')
    ax.legend()
    plt.show()


def Scatter_objectives(objectives_by_generations):
    '''
    # Scatter objectives by generations
    '''
    # Number of generation to plot
    num_gen_plot = 10
    num_generations = len(objectives_by_generations)
    step = num_generations/num_gen_plot
    # Choose a colormap (replace 'viridis' with your preferred choice)
    cmap = plt.cm.winter
    # Create an array of values from 0 to 100 (one for each point)
    values = np.linspace(0, 1, num_gen_plot)
    # Generate the list of colors using the colormap
    color = cmap(values)

    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot(projection='3d')
    for i in range(len(objectives_by_generations)):
        if(i==0):
            f = []
            for individual in objectives_by_generations[i]:
                f.append([individual[0], individual[1], individual[2]])
            f = np.array(f)
            ax.scatter(f[:,0],f[:,1], f[:,2], c = color[int(i/step)],label=f'Gen #{i+1}')
        elif((i+1)%step==0):
            f = []
            for individual in objectives_by_generations[i]:
                f.append([individual[0], individual[1], individual[2]])
            f = np.array(f)
            ax.scatter(f[:,0],f[:,1], f[:,2], c = color[int((i+1)/step)-1],label=f'Gen #{i+1}')
    # ax.view_init(azim=45,elev = 30)
    ax.set_xlabel('Power Consumption')
    ax.set_ylabel('Number of Active Sensors')
    ax.set_zlabel('Active Sensors avg. distance to Sink Node ')
    plt.legend()
    plt.title('Objective funtions value over generations')
    plt.show()


def Plot_fitness(num_sensors, num_results, name_pattern='best_indi_fitness', name_dataset = 'dataset_0', length=1000, width=50, distribution='uniform'):
    '''
    # Plot fitness of one dataset with mean and standard deviation
    Prerequisites: Results available
    
    Arguments:
        num_sensors: Number of sensors
        num_results: Number of fitness files
        name_pattern: Pattern of result file name, eg: 'best_indi_fitness' 
        name_dataset: Name of the folder contains fitness files
    '''
    fitneses = []
    for i in range(num_results):
        try:
            with open(f'MOEAD_Results/{distribution}/{width}x{length}unit/{num_sensors}sensors/{name_dataset}/{name_pattern}_{i}.pickle','rb') as file:
                fitneses.append(pickle.load(file))
        except FileNotFoundError:
            print('FileNotFoundError')
            print(f'Results/{distribution}/{width}x{length}unit/{num_sensors}sensors/{name_pattern}_{i}.pickle not existed')
            return
    
    fitneses = np.array(fitneses)
    mean = np.mean(fitneses,axis=0)
    std = np.std(fitneses,axis=0)
    
    fix, ax = plt.subplots()
    x = np.arange(len(fitneses[0]))
    ax.plot(x,mean)
    ax.fill_between(x, mean-std, mean+std, alpha =0.2)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    # ax.set_ylim(1,10)
    plt.title(f'{name_pattern} Mean and Standard deviation, {num_sensors} sensors, {distribution} distribution')
    plt.savefig('scene.svg')
    plt.show()

def Compare_objectives(moead_objectives,nsga2_objectives):
    '''
    # Compare output of MOEAD and NSGA2
    '''

    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot(projection='3d')
    moead_objectives = np.array(moead_objectives)
    nsga2_objectives = np.array(nsga2_objectives)


    ax.scatter(moead_objectives[:,0],moead_objectives[:,1],moead_objectives[:,2],c='red',label='MOEA/D')
    ax.scatter(nsga2_objectives[:,0],nsga2_objectives[:,1],nsga2_objectives[:,2],c='blue',label='NSGA-II')

    ax.set_xlabel('Power Consumption')
    ax.set_ylabel('Number of Active Sensors')
    ax.set_zlabel('Active Sensors avg. distance to Sink Node ')
    plt.legend()
    plt.title('Objective funtions value over generations')
    plt.show()

def Open(link):
    try:
        with open(link,'rb') as file:
            obj = pickle.load(file)

            return obj
    except:
        print('Wrong link or file not exist!')
    
def Scatter_objectives_PV(objectives_by_generations):
    '''
    Scatter objectives by generations with PyVista
    
    Return a PyVista.Plotter
    '''
    plotter = pv.Plotter()
    
    obj = np.array(objectives_by_generations)
    max_obj = [np.max(obj[:,:,i]) for i in range(3)]
    plotter.set_scale(xscale=max_obj[1]/max_obj[0],zscale=max_obj[1]/max_obj[2])
    
    # Add Ox, Oy, Oz axes
    plotter.add_mesh(pv.Line((0,0,0),(max_obj[0]*1.2,0,0)),line_width=5,color='black')
    plotter.add_mesh(pv.Line((0,0,0),(0,max_obj[1]*1.2,0)),line_width=5,color='black')
    plotter.add_mesh(pv.Line((0,0,0),(0,0,max_obj[2]*1.2)),line_width=5,color='black')

    # Number of generation to plot
    num_gen_plot = 10
    num_generations = len(objectives_by_generations)
    step = num_generations/num_gen_plot
    # Choose a colormap (replace 'viridis' with your preferred choice)
    cmap = plt.cm.winter
    # Create an array of values from 0 to 100 (one for each point)
    values = np.linspace(0, 1, num_gen_plot)
    # Generate the list of colors using the colormap
    color = cmap(values)

    for i in range(len(objectives_by_generations)):
        points = pv.PolyData(obj[i])
        if(i==0):
            plotter.add_points(
                    points,
                    style='points',
                    color=color[int(i/step)],
                    point_size=8,
                    label=f'Gen #{i+1}',
                    render_points_as_spheres=True)
            
        elif((i+1)%step==0):
            plotter.add_points(
                    points,
                    style='points',
                    color=color[int(i/step)],
                    point_size=8,
                    label=f'Gen #{i+1}',
                    render_points_as_spheres=True)
    plotter.add_legend()
    plotter.show()
    return plotter
    

def Scatter_objectives_Plotly(objectives_by_generations):
    '''
    Scatter objectives by generations with plotly
    
    '''
    # Number of generation to plot
    num_gen_plot = 10
    num_generations = len(objectives_by_generations)
    step = num_generations/num_gen_plot
    # Choose a colormap (replace 'viridis' with your preferred choice)
    cmap = plt.cm.winter
    # Create an array of values from 0 to 100 (one for each point)
    values = np.linspace(0, 1, num_gen_plot)
    # Generate the list of colors using the colormap
    color = cmap(values)

    df = pd.DataFrame(columns=['Power Consumption','Number of Active Sensors','Active Sensors avg. distance to Sink Node','Generation No.','color'])
    for i in range(len(objectives_by_generations)):
        if(i==0):
            for indi in objectives_by_generations[i]:
                df.loc[len(df)] = [indi[0],indi[1],indi[2],i+1,color[int((i+1)/step)]]
        elif((i+1)%step==0):
            for indi in objectives_by_generations[i]:
                df.loc[len(df)] = [indi[0],indi[1],indi[2],i+1,color[int((i+1)/step)-1]]
    
    df = df[df['Generation No.']>98]
    fig = px.scatter_3d(df, x='Power Consumption', y='Number of Active Sensors', z='Active Sensors avg. distance to Sink Node',
                color='Generation No.')
    fig.update_traces(marker=dict(size=3))
    fig.show()
    fig.write_html('scene.html')

def Compare_objectives_Plotly(moead_objectives,nsga2_objectives):
    '''
    # Compare output of MOEAD and NSGA2 with Plotly
    '''

    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot(projection='3d')
    moead_objectives = np.array(moead_objectives)
    nsga2_objectives = np.array(nsga2_objectives)

    df = pd.DataFrame(columns=['Power Consumption','Number of Active Sensors','Active Sensors avg. distance to Sink Node','Algorithm'])

    moead_last_gen, nsga_last_gen = moead_objectives[-1], nsga2_objectives[-1]

    for indi in moead_last_gen:
        df.loc[len(df)] = [indi[0],indi[1],indi[2],'MOEA/D']
    for indi in nsga_last_gen:
        df.loc[len(df)] = [indi[0],indi[1],indi[2],'NSGA']

    fig = px.scatter_3d(df, x='Power Consumption', y='Number of Active Sensors', z='Active Sensors avg. distance to Sink Node',
                color='Algorithm')
    fig.update_traces(marker=dict(size=3))
    fig.show()
    fig.write_html('Compare objectives.html')

def Plot_hyper_volume(objectives_by_generations):
    '''
    # Plot hyper volume by generations
    
    Arguments: 
        objectives_by_generations: list of shape [num. generations, pop. size, 3]
    '''
    max_f1 = np.max(np.array(objectives_by_generations)[:,:,0])
    max_f2 = np.max(np.array(objectives_by_generations)[:,:,1])
    max_f3 = np.max(np.array(objectives_by_generations)[:,:,2])

    hyper_volume = []
    for i in range(len(objectives_by_generations)):
        hyper_volume.append(pg.hypervolume(objectives_by_generations[i]).compute([max_f1,max_f2,max_f3]))
    plt.plot(hyper_volume)
    plt.show()
    

def Compare_Set_Coverage():
    moead_dom_nsga2 = []
    nsga2_dom_moead = []
    
    def dominate(obj1, obj2):
        obj1, obj2 = np.array(obj1), np.array(obj2)
        smaller_or_equal = obj1 <= obj2
        smaller = obj1 < obj2
        if np.all(smaller_or_equal) and np.any(smaller):
            return True

        return False
    
    for num_sensor in [100,300,700]:
        moead_last_gen = Open(f'MOEAD_Results/uniform/50x1000unit/{num_sensor}sensors/dataset_0/objectives_by_generations_0.pickle')[-1]
        nsga2_last_gen = Open(f'NSGA2_Results/uniform/50x1000unit/{num_sensor}sensors/dataset_0/objectives_by_generations_0.pickle')[-1]

        count_moead_dom, count_nsga2_dom = 0,0 
        for o1 in nsga2_last_gen:
            for o2 in moead_last_gen:
                if(dominate(o1,o2)):
                    count_nsga2_dom+=1
                elif(dominate(o2,o1)):
                    count_moead_dom+=1

        moead_dom_nsga2.append(count_moead_dom/len(moead_last_gen))
        nsga2_dom_moead.append(count_nsga2_dom/len(nsga2_last_gen))
    
    plt.ylim((0,1))
    plt.plot([100,300,700],moead_dom_nsga2,label='MOEA/D > NSGA2',marker='*',ms=10)
    plt.plot([100,300,700],nsga2_dom_moead,label='NSGA2 > MOEA/D',marker='*',ms=10)
    plt.legend()
    plt.xlabel('Number of sensors')
    plt.ylabel('Set coverage')
    plt.title('Compare Set coverage')
    plt.show()

def Compare_Distance_to_Reference():
    d_ref_moead = []
    d_ref_nsga2 = []
    for num_sensor in [100,300,700]:
        moead_last_gen = Open(f'MOEAD_Results/uniform/50x1000unit/{num_sensor}sensors/dataset_0/objectives_by_generations_0.pickle')[-1]
        nsga2_last_gen = Open(f'NSGA2_Results/uniform/50x1000unit/{num_sensor}sensors/dataset_0/objectives_by_generations_0.pickle')[-1]

        max_f1 = np.max(np.array(moead_last_gen+nsga2_last_gen)[:,0])
        max_f2 = np.max(np.array(moead_last_gen+nsga2_last_gen)[:,1])
        max_f3 = np.max(np.array(moead_last_gen+nsga2_last_gen)[:,2])
        reference_point = np.array([max_f1,max_f2,max_f3])

        current_d_ref_moead = 0
        current_d_ref_nsga2 = 0

        for i in range(len(moead_last_gen)):
            current_d_ref_moead += np.sqrt(np.sum(np.square(moead_last_gen[i]-reference_point)))
            current_d_ref_nsga2 += np.sqrt(np.sum(np.square(nsga2_last_gen[i]-reference_point)))

        d_ref_moead.append(current_d_ref_moead)
        d_ref_nsga2.append(current_d_ref_nsga2)

    plt.plot([100,300,700],d_ref_moead,label='MOEA/D',marker='*',ms=10)
    plt.plot([100,300,700],d_ref_nsga2,label='NSGA2/D',marker='*',ms=10)
    plt.xlabel('Number of sensors')
    plt.ylabel('Distance')
    plt.title('Compare Distance to Reference point')
    plt.legend()
    plt.show()