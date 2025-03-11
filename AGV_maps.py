import csv
import os
import shutil
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
from fealpy.backend import backend_manager as bm

bm.set_backend('pytorch')


class AGVProblem:
    def __init__(self, MAP, start, goal, coords):
        self.MAP = MAP
        self.start = start
        self.goal = goal
        self.coords = coords
        self.data = {}
        self.precomputed_paths = None  # 预计算路径

        # 初始化时计算距离矩阵
        self.data['node'] = self.coords
        self.data['D'] = squareform(pdist(self.data['node']))
    
    def calD(self):
        """计算距离矩阵和网络图"""
        self.data['R'] = 1 
        self.data['numLM0'] = 1 
        self.data['trun_weight'] = 0.5 
        
        p1, p2 = bm.where(bm.array(self.data['D']) <= bm.array(self.data['R'])) 
        D = self.data['D'][(p1, p2)].reshape(-1, 1) 
        self.data['net'] = sp.csr_matrix((D.flatten(), (p1, p2)), shape = self.data['D'].shape) # 构建邻接矩阵
        self.data['G'] = nx.DiGraph(self.data['net']) # 构建有向图 
        
        # self.data['node'] = self.data['node'][:, [1, 0]] 
        self.data['noS'] = bm.where((self.data['node'][:, 0] == self.start[0]) & (self.data['node'][:, 1] == self.start[1]))[0][0] 
        self.data['noE'] = bm.where((self.data['node'][:, 0] == self.goal[0]) & (self.data['node'][:, 1] == self.goal[1]))[0][0] 

        # 预计算所有节点对之间的最短路径
        if self.precomputed_paths is None:
            self.precomputed_paths = dict(nx.all_pairs_shortest_path_length(self.data['G']))
        
        return self.data   
    
    def gbestroute(self, X):
        """根据输入种群 X 计算最优路径"""
        sorted_numbers = bm.argsort(X)
        NP = sorted_numbers.shape[0]

        paths = []
        result = {}
        sorted_numbers_flat = sorted_numbers[:, :self.data['numLM0']]
        noS = bm.full((NP,), self.data['noS']).reshape(-1, 1)
        noE = bm.full((NP,), self.data['noE']).reshape(-1, 1)
        path0 = bm.concatenate((noS, sorted_numbers_flat, noE), axis=1) # 构建路径
        distances = bm.zeros((NP, path0.shape[1] - 1))
        # 使用预计算的最短路径
        for j in range(NP):
            for i in range(path0.shape[1] - 1):
                source = int(path0[j][i])
                target = int(path0[j][i + 1])
                distances[j, i] = self.precomputed_paths[source].get(target, float('inf'))  # 查找预计算路径

            path = []
            for i in range(path0.shape[1] - 1):
                source = int(path0[j][i])
                target = int(path0[j][i + 1])
                path.extend(nx.shortest_path(self.data['G'], source=source, target=target))  # 获取路径节点       
            pathxy = [x for x, y in zip(path, path[1:] + [None]) if x != y]  # 去除重复节点
            paths.append(pathxy)
            
        turn_penalties = self.check_turn(paths) # 计算转弯惩罚
        fit = bm.sum(distances, axis=1)
        fit  = fit + self.data['trun_weight'] * turn_penalties

        result['path'] = paths[0]
        result['truns'] = turn_penalties[0]
        result['fit'] = fit
        return result 

    def fitness(self, X):
        """计算路径的适应度值"""
        result = self.gbestroute(X)
        fit = result['fit']
        return fit
    
    def check_turn(self, paths):
        """计算路径中的转弯次数"""
        turn_penalties = bm.zeros((len(paths),)) 
        for idx, path in enumerate(paths):
            count_turn = 0
            for i in range(0, len(path) - 2):
                node1xy, node3xy = self.data['node'][path[i]], self.data['node'][path[i + 2]]
                if (node1xy[1] != node3xy[1]) and (node1xy[0] != node3xy[0]):
                    count_turn += 1
            turn_penalties[idx] = count_turn
        return turn_penalties

def print_results_short(results):
    """打印并保存优化结果"""
    for algo_name, result in results.items():
        print('The best optimal value by {0:5} \t'
            'is :{1:12.4f}\t'
            'trun_counts :{2:12}\t'
            'Time :{3:8.4f} seconds'.format(algo_name, result['gbest_f'], result['truns'], result['time']))
    
    with open('results_Shortest.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Algorithm', 'Best Optimal Value', 'trun_counts', 'Time', 'Grid paths']
        writer.writerow(header)
        for algo_name, result in results.items():
            writer.writerow([algo_name, result['gbest_f'], result['truns'], result['time'], result['route']])
    print("\nResults have been successfully saved to 'results_Shortest.csv'.") 

def printroute_short(MAP, results, save_path="images_Shortest"):  
    """绘制并保存路径图像"""
    MAPs = MAP['maps']
    coords = MAP['coords']
    alg_name = list(results.keys())
    path_len = [len(result['route']) for result in results.values()]
    route_all = bm.array([result['route'] for result in results.values()], dtype=object) if len(set(path_len)) != 1 else bm.array([result['route'] for result in results.values()]) # 确保路径数据一致性

    b = MAPs.shape
    start = coords[route_all[0][0]]
    goal = coords[route_all[0][-1]]
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    for j in range(route_all.shape[0]):
        path = route_all[j]
        plt.figure()
        plt.scatter(start[1], start[0], color = 'blue', s = 50, label="Start")
        plt.scatter(goal[1], goal[0], color = 'green', s = 50, label="Goal")

        dense_MAPs = MAPs.toarray() # 绘制地图
        plt.imshow(dense_MAPs, cmap = 'gray', origin='upper')

        xx = bm.linspace(0, b[1], b[1]) - 0.5
        for i in range(0, b[0]):
            plt.plot(xx, bm.full_like(xx, i) - 0.5, '-', color='gray')
        yy = bm.linspace(0, b[0], b[0]) - 0.5
        for i in range(0, b[1]):
            plt.plot(bm.full_like(yy, i) - 0.5, yy, '-', color='gray')

        xpath = coords[path, 0]
        ypath = coords[path, 1]
        plt.plot(ypath, xpath, '-', color = 'red')
        plt.plot([xpath[-1], goal[0]], [ypath[-1], goal[1]], '-', color = 'red')
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Optimal Paths from {alg_name[j]}')
        
        figure_filename = f"{alg_name[j]}_optimal_path.png"
        plt.savefig(os.path.join(save_path, figure_filename), dpi=100)
        plt.close()
    print("Images have been successfully saved to 'images_Shortest'file.")



if __name__ == "__main":
    pass

