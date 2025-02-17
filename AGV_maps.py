import csv
import os
import shutil
import matplotlib.pyplot as plt
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from fealpy.backend import backend_manager as bm


def process_and_save_map(map_func, save_dir):
    """将二维0,1数组转换为. T @形式的数据并保存"""
    original_map = map_func()
    target_height = original_map.shape[0]
    target_width = original_map.shape[1]
    target_map = [['' for _ in range(target_width)] for _ in range(target_height)]
    
    for y in range(target_height):          # 进行 0 到 T、1 到 @ 的转换
        for x in range(target_width):
            if original_map[y][x] == 0:
                target_map[y][x] = 'T'
            else:
                target_map[y][x] = '@'
    
    for y in range(1, target_height - 1):  # 遍历目标地图，将被 T 四周包围的 T 改为 .
        for x in range(1, target_width - 1):
            if target_map[y][x] == 'T':
                top = target_map[y - 1][x] == 'T'
                bottom = target_map[y + 1][x] == 'T'
                left = target_map[y][x - 1] == 'T'
                right = target_map[y][x + 1] == 'T'
                if top and bottom and left and right:
                    target_map[y][x] = '.'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    func_name = map_func.__name__
    file_name = f"0_{func_name}.map"
    save_path = os.path.join(save_dir, file_name)
 
    with open(save_path, 'w') as f:
        f.write("type octile\n")
        f.write(f"height {target_height}\n")
        f.write(f"width {target_width}\n")
        f.write("map\n")
        for row in target_map:
            f.write(''.join(row) + '\n')
    print(f"地图已保存到 {save_path}")


def short_folder(file_path):
    """读取地图文件并解析为稀疏矩阵，返回地图信息、起点和终点"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    file_type = None
    height = None
    width = None
    map_data = []
    for line in lines:
        line = line.strip()
        if line.startswith('type'):
            file_type = line.split()[1]
        elif line.startswith('height'):
            height = int(line.split()[1])
        elif line.startswith('width'):
            width = int(line.split()[1])
        elif line.startswith('map'):
            continue  # 跳过 'map' 标题行
        else:
            map_row = [0 if char == '@' else 1 for char in line] # '@' 表示障碍物，其他为可行区域
            map_data.append(map_row)
    
    map_array = bm.array(map_data)
    sparse_map = csr_matrix(map_array) # 将地图转换为稀疏矩阵

    rows, cols = sparse_map.nonzero()
    coords = list(zip(rows, cols))
    start = coords[0]  # 第一个点
    goal = coords[-1]  # 最后一个点
    
    return {
        'height': height,
        'width': width,
        'maps': sparse_map,
        'start': start,
        'goal': goal
    }

class AGVProblem:
    def __init__(self, MAP, start, goal):
        self.MAP = MAP
        self.start = start
        self.goal = goal
        self.data = {}
        self.precomputed_paths = None  # 预计算路径
    
    def calD(self):
        """计算距离矩阵和网络图"""
        self.data['R'] = 1 # 邻域半径
        self.data['numLM0'] = 1 # 路径中的关键点数量
        self.data['trun_weight'] = 0.5 # 转弯惩罚权重

        rows, cols = self.MAP.nonzero()
        node = [[col, row] for row, col in zip(rows, cols)]
        self.data['node'] = bm.array(node)

        # self.data['D'] = squareform(pdist(self.data['node'])) 
                # 预计算距离矩阵
        if not hasattr(self, 'D'):
            self.data['D'] = squareform(pdist(self.data['node']))

        p1, p2 = bm.where(bm.array(self.data['D']) <= bm.array(self.data['R'])) 
        D = self.data['D'][(p1, p2)].reshape(-1, 1) 
        self.data['net'] = sp.csr_matrix((D.flatten(), (p1, p2)), shape = self.data['D'].shape) # 构建邻接矩阵
        self.data['G'] = nx.DiGraph(self.data['net']) # 构建有向图 

        self.data['node'] = self.data['node'][:, [1, 0]] 
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
            # for j in range(0, NP): 
            #     distance = [nx.shortest_path_length(self.data['G'], source=int(path0[j][i]), target=int(path0[j][i + 1]), weight=None) for i in range(path0.shape[1] - 1)]
            #     distances[j, :] = bm.tensor(distance)

            #     path = [nx.shortest_path(self.data['G'], source=int(path0[j][i]), target=int(path0[j][i + 1])) for i in range(path0.shape[1] - 1)] # 获取路径节点
            #     combined_list = []
            #     for sublist in path:
            #         combined_list.extend(sublist)
            #     pathxy = [x for x, y in zip(combined_list, combined_list[1:] + [None]) if x != y] # 去除重复节点
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

def printroute_short(MAPs, results, save_path="images_Shortest"):  
    """绘制并保存路径图像"""
    alg_name = list(results.keys())
    path_len = [len(result['route']) for result in results.values()]
    route_all = bm.array([result['route'] for result in results.values()], dtype=object) if len(set(path_len)) != 1 else bm.array([result['route'] for result in results.values()]) # 确保路径数据一致性

    rows, cols = MAPs.nonzero()
    nodes = bm.array([[row, col] for row, col in zip(rows, cols)])

    b = MAPs.shape
    start = nodes[route_all[0][0]]
    goal = nodes[route_all[0][-1]]
    
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

        xpath = nodes[path, 0]
        ypath = nodes[path, 1]
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

