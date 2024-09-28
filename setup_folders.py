import os

if __name__ == "__main__":
    parrent_dir = os.getcwd()

    for alg in ['NSGA2_Results', 'MOEAD_Results']:
        p1 = os.path.join(parrent_dir, alg)
        for dist in ['uniform', 'gauss', 'exponential']:
            p2 = os.path.join(p1, dist)
            for  size in ['50x1000unit', '500x1000unit']:
                p3 = os.path.join(p2, size)
                for num_sensor in ['100sensors', '300sensors', '700sensors']:
                    p4 = os.path.join(p3, num_sensor)
                    for dataset in range(5):
                        p5 = os.path.join(p4, f'dataset_{dataset}')
                        
                        try:
                            os.makedirs(p5)
                        except FileExistsError: 
                            print(f'Folder {p5} existed!')
                            continue