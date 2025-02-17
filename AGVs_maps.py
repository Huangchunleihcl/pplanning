import os
import csv
from fealpy.backend import backend_manager as bm
import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import label
import shutil
import imageio
from itertools import chain
import matplotlib.pyplot as plt


def grid_random(seed=None):
    dim = 20
    obstacle_prob = 0.1
    bm.random.seed(seed)
    Map = bm.random.choice([0, 1], size=(dim, dim), p=[1 - obstacle_prob, obstacle_prob])
    return Map

AGV_data = [
    {
        'map': grid_random(seed=42),
        'num' : 2,
        'start': [[0, 0],[2, 17]],
        'goal': [[19, 19],[14, 6]],
        'dim': 20,
    },
]

'''
来源:https://movingai.com/benchmarks/street/index.html

使用 gridworld 域作为基准，这里使用 2D 网格世界基准测试
城市/街道地图由 Konstantin Yakovlev 和 Anton Andreychuk提供。
'''



class AGVsProblem:
    def __init__(self, MAP, starts, goals, num_agvs, dim):
        self.MAP = MAP
        self.starts = starts
        self.goals = goals
        self.num_agvs = num_agvs
        self.dim = dim
        self.data = {}

    def calD(self):
        self.data["R"] = 1 # 每步最长距离
        self.data['numLM0'] = 1 #一次最大步数
        self.data['collision_weight'] = self.dim # 碰撞权重
        self.data['trun_weight'] = 0.2 # 转弯权重

        L, _ = label(self.MAP) 
        indices = bm.where(bm.array(L) > 0) 
        landmark = bm.concatenate([bm.array(L[i]) for i in indices]) 
        self.data['landmark'] = bm.array(landmark)

        node = [[j, i] for i in range(self.MAP.shape[0]) for j in range(self.MAP.shape[1]) if self.MAP[i, j] == 0]
        self.data['node'] = bm.array(node)
        self.data['D'] = squareform(pdist(self.data['node']))
        p1, p2 = bm.where(bm.array(self.data['D']) <= bm.array(self.data['R']))
        D = self.data['D'][(p1, p2)].reshape(-1, 1)
        self.data['net'] = sp.csr_matrix((D.flatten(), (p1, p2)), shape = self.data['D'].shape)
        self.data['G'] = nx.DiGraph(self.data['net'])

        self.data['node'] = self.data['node'][:, [1, 0]]
        self.data['noS'] = bm.nonzero(bm.array([bm.all(self.data['node'] == start, axis=1) for start in self.starts]))[1]
        self.data['noE'] = bm.nonzero(bm.array([bm.all(self.data['node'] == goal, axis=1) for goal in self.goals]))[1]
        return self.data
    
    def fitness(self, X):
        sorted_numbers = bm.argsort(X)
        NP = sorted_numbers.shape[0]

        paths = []
        fit = bm.zeros((NP,))
        for idx in range(0, NP):
            gbest = sorted_numbers[idx]
            result = self.gbestroute(gbest)
            fit[idx] = result['fit']
            paths.append(result['path'])

        turn_penalties = self.check_turn(paths)
        collision_penalties = self.check_collision(paths)
        fit = fit + self.data['trun_weight'] * turn_penalties + self.data['collision_weight'] * collision_penalties
        return fit

    def check_turn(self, paths):
        turn_penalties = bm.zeros((len(paths),)) 
        for idx, path in enumerate(paths):
            count_turn = 0
            for path_agv in path:
                for i in range(0, len(path_agv) - 2):
                    node1xy, node3xy = self.data['node'][path_agv[i]], self.data['node'][path_agv[i + 2]]
                    if node1xy[1] != node3xy[1] and node1xy[0] != node3xy[0]:
                        count_turn += 1
            turn_penalties[idx] = count_turn
        return turn_penalties
  
    def check_collision(self, paths):
        collision_penalties = bm.zeros((len(paths),))
        for idx, path in enumerate(paths):
            count_collision = 0
            for colli in range(len(path)):
                for collj in range(colli + 1, len(path)):
                    path_colli = path[colli]
                    path_collj = path[collj]
                    check_path = max(len(path_colli), len(path_collj))
                    path_colli += [path_colli[-1]] * (check_path - len(path_colli))
                    path_collj += [path_collj[-1]] * (check_path - len(path_collj))
                    for i in range(0, check_path - 1):
                        if path_colli[i] == path_collj[i] or (path_colli[i], path_colli[i + 1]) == (path_collj[i + 1], path_collj[i]):
                            count_collision += 1
            collision_penalties[idx] = count_collision
        return collision_penalties
    
    def gbestroute(self, sorted_numbers):
        route = {}
        paths_all = []
        path0_all = []
        distances_all = []
        
        for agv in range(self.num_agvs):
            sorted_numbers_agv = sorted_numbers[(agv * self.data['numLM0']): ((agv + 1) * self.data['numLM0'])]
            sorted_numbers_agv = [element for element in sorted_numbers_agv]
            sorted_path = [self.data['noS'][agv]] + sorted_numbers_agv + [self.data['noE'][agv]]
            path0_all.append(sorted_path)

        for agv, path_agv in enumerate(path0_all):
            paths = []
            distances = []
            for i in range(0, len(path_agv) - 1):
                source = int(path_agv[i])
                target = int(path_agv[i + 1])
                path = nx.shortest_path(self.data['G'], source = source, target = target)
                distance = nx.shortest_path_length(self.data['G'], source = source, target = target, weight = None)  
                distances.append(distance)
                paths.append(path)

            combined_list = []
            for path_i in paths:
                combined_list.extend(path_i)
            path0 = [x for x, y in zip(combined_list, combined_list[1:] + [None]) if x != y] 

            paths_all.append(path0)
            distances_all.append(distances)

        flattened_distances = chain(*distances_all)
        fit = sum(flattened_distances)

        route['path'] = paths_all
        route['fit'] = fit
        return route
    
def print_results_ob(results):
    with open('results_OA.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Algorithm', 'Best Optimal Value', 'Time', 'AGV(start,goal)', 'fit_agv', 'Route']
        writer.writerow(header)

        for algo_name, result in results.items():
            print(f'Multi-AGVs is calculated based on {algo_name}')
            time = result['time']
            gbest_f = result['gbest_f']
            results = result['AGV']
            for agv_id, agv_result in results.items():
                fit_agv = agv_result['fit_agv']
                route = agv_result['route']
                agv_name = [route[0],route[-1]]

                writer.writerow([algo_name, gbest_f, time, agv_name, fit_agv, route])

                print(f'{agv_id}   fit: {fit_agv:6.2f}   Route: {route}')
            print(f'gbest_f: {gbest_f:6.2f}   Total Time : {time:8.4f} seconds\n')
    print("Results have been successfully saved to 'results_OA.csv'.")

def printroute_ob(MAPs, results, save_dir='images_OA'):
    nodes = bm.array([[i, j] for i in range(MAPs.shape[0]) for j in range(MAPs.shape[1]) if MAPs[i, j] == 0])
    b = MAPs.shape
    MAPs = 1 - MAPs

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir) 
    os.makedirs(save_dir)

    for algo_name, agv_data in results.items():
        algo_folder = os.path.join(save_dir, algo_name)
        if not os.path.exists(algo_folder):
            os.makedirs(algo_folder)

        max_length = max(len(agv_info['route']) for agv_info in agv_data['AGV'].values())
        used_colors = {agv_name: bm.random.rand(3,) for agv_name in agv_data['AGV'].keys()}
        
        GIF_files = []
        for step in range(max_length):
            plt.figure()
            plt.imshow(MAPs[::1], cmap = 'gray')

            xx = bm.linspace(0,b[1],b[1]) - 0.5
            yy = bm.zeros(b[1]) - 0.5
            for i in range(0, b[0]):
                yy = yy + 1
                plt.plot(xx, yy,'-',color = 'gray')
            x = bm.zeros(b[0])-0.5
            y = bm.linspace(0,b[0],b[0])-0.5
            for i in range(0, b[1]):
                x = x + 1
                plt.plot(x,y,'-',color = 'gray')
            
            for agv_name, agv_info in agv_data['AGV'].items():
                path = agv_info['route']
                agv_color = used_colors[agv_name]

                extended_path = path[:step + 1]
                coords = bm.array([nodes[i] for i in extended_path])
                if coords.size > 0:
                    plt.plot(coords[:, 1], coords[:, 0], '-', color=agv_color, markersize=30)

                start_node = nodes[path[0]]
                end_node = nodes[path[-1]]
                plt.scatter(start_node[1], start_node[0], color = agv_color, s = 50, label=f'agv {agv_name} = {agv_info["fit_agv"]}')
                plt.scatter(end_node[1], end_node[0], color = agv_color, s = 50)

            plt.xticks([])
            plt.yticks([])
            plt.title(f"Best optimal value of {algo_name} = {agv_data['gbest_f']} \n Step {step + 1}/{max_length}")
            plt.legend(loc='upper right', bbox_to_anchor=(1.36, 1))

            filename = f"step_{step + 1:03d}.png"
            plt.savefig(os.path.join(algo_folder, filename))
            plt.close()
    
            GIF_files.append(os.path.join(algo_folder, filename))
        gif_filename = os.path.join(algo_folder, f"{algo_name}.gif")
        with imageio.get_writer(gif_filename, mode='I', fps = 3) as writer:
            for GIF_file in GIF_files:
                image = imageio.imread(GIF_file)
                writer.append_data(image)
    print("Images have been successfully saved to 'images_OA'file.\n")


