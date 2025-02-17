from fealpy.backend import backend_manager as bm
import matplotlib.pyplot as plt
import os
import csv
import shutil

def read_tsp_file(file_path):
    TSP_data = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        current_info = {}
        start_reading_coords = False
        city_coords = []
        for line in lines:
            line = line.strip()
            if line.startswith("NAME"):
                name = line.split(':')[1].strip()
                current_info["name"] = name
            elif line.startswith("DIMENSION"):
                dimension = int(line.split(':')[1].strip())
                current_info["dimension"] = dimension
            elif line.startswith("NODE_COORD_SECTION") or line.startswith("DISPLAY_DATA_SECTION"):
                start_reading_coords = True
                continue
            elif line == "EOF":
                current_info["citys"] = bm.array(city_coords)
                TSP_data.append(current_info)
                break
            elif start_reading_coords:
                parts = line.split()
                x = float(parts[1])
                y = float(parts[2])
                city_coords.append([x, y])
    return TSP_data

def tsp_folder(folder_path):
    all_tsp_data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.tsp'):
                file_full_path = os.path.join(root, file)
                tsp_data = read_tsp_file(file_full_path)
                all_tsp_data.extend(tsp_data)
    return all_tsp_data

class TSProblem:
    def __init__(self, citys) -> None:
        self.citys = citys
        self.D = bm.zeros((citys.shape[0], citys.shape[0]))

    def calD(self):
        n = self.citys.shape[0] 
        diff = self.citys[:, None, :] - self.citys[None, :, :]
        self.D = bm.sqrt(bm.sum(diff ** 2, axis = -1))
        self.D[bm.arange(n), bm.arange(n)] = 2.2204e-16

    def fitness(self, x):
        index = bm.argsort(x, axis=-1)
        distance = self.D[index[:, -1], index[:, 0]]
        for i in range(x.shape[1] - 1):
            dis = self.D[index[:, i], index[:, i + 1]]
            distance = distance + dis
        return distance

def print_results_tsp(results):
    for algo_name, result in results.items():
        print('The best optimal value by {0:5} \t'
            'is :{1:12.4f}\t'
            'Time :{2:8.4f} seconds'.format(algo_name, result['gbest_f'], result['time']))
    
    with open('results_TSP.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Algorithm', 'Best Optimal Value', 'Time', 'Citys path']
        writer.writerow(header)
        for algo_name, result in results.items():
            writer.writerow([algo_name, result['gbest_f'], result['time'], result['route']])
    print("\nResults have been successfully saved to 'results_TSP.csv'.")           


def printroute_tsp(citys, results, save_path="images_TSP"):
    route_citys_all = bm.array([result['route_citys'].tolist() for result in results.values()])
    alg_name = list(results.keys())

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    dim_citys = citys.shape[1]
    if dim_citys == 2:
        for i in range(route_citys_all.shape[0]):
            plt.figure()
            plt.title(f"{alg_name[i]} optimal path")
            plt.scatter(citys[:, 0], citys[:, 1])
            plt.plot(route_citys_all[i, :, 0], route_citys_all[i, :, 1])

            figure_filename = f"{alg_name[i]}_optimal_path.png"
            plt.savefig(os.path.join(save_path, figure_filename))
            plt.close()

    elif dim_citys == 3:
        for i in range(route_citys_all.shape[0]):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(f"{alg_name[i]} optimal path")
            ax.scatter(citys[:, 0], citys[:, 1], citys[:, 2]) 
            ax.plot(route_citys_all[i, :, 0], route_citys_all[i, :, 1], route_citys_all[i, :, 2])

            figure_filename = f"{alg_name[i]}_optimal_path.png"
            plt.savefig(os.path.join(save_path, figure_filename))
            plt.close() 
    print("Images have been successfully saved to 'images_TSP'file.") 




if __name__ == "__main":
    pass