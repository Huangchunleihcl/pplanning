import os
import logging
import json
import csv
from tqdm import tqdm
import networkx as nx
from fealpy.sparse.csr_tensor import CSRTensor
from fealpy.backend import backend_manager as bm
from fealpy.backend import TensorLike

bm.set_backend('pytorch')
# device = 'cuda'
# bm.set_default_device(device)


class NumpyEncoder(json.JSONEncoder):
    """处理pytorch数据类型的JSON编码"""
    def default(self, obj):
        if isinstance(obj, TensorLike):
            return obj.tolist()
        elif isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, float):
            return int(obj)
        return super().default(obj)
    

class MapDataPreprocessor:
    def __init__(self, open_dir='output_maps'):
        self.open_dir = open_dir


    def save_to_json(self, file_name='features1.json'):
        """将结果保存为JSON文件"""
        output_file = os.path.join(self.open_dir, file_name)
        features= self.extract_features()
        try:
            with open(output_file, 'w') as f:
                json.dump(features, f, indent=4, cls=NumpyEncoder)
            print(f"结果已成功保存到 {output_file}")
        except Exception as e:
            logging.error(f"保存结果到 JSON 文件时出错: {str(e)}")


    def extract_features(self):
        """提取地图特征"""
        features = {}
        cases_list = self.parse_all_maps()
        for case in tqdm(cases_list, desc="提取地图特征", unit="地图"):
            case_id = case["case_id"]
            map_info = case["map_data"]
            points = map_info['coords']
            height, width = map_info['height'], map_info['width']

            features[case_id] = {}
            total_pixels = height * width
            free_pixels = len(points)

            features[case_id]["obs_density"] = round(1 - (free_pixels / total_pixels), 4)
            features[case_id]["manhattan_dist"] = abs(map_info['start'][0] - map_info['goal'][0]) + abs(map_info['start'][1] - map_info['goal'][1])

            G = nx.Graph()
            G.add_nodes_from(points)
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右/下/左/上
            for x, y in points:
                for dx, dy in directions:
                    nx_, ny = x + dx, y + dy
                    if (nx_, ny) in points:
                        G.add_edge((x, y), (nx_, ny))
            features[case_id]["nodes"] = G.number_of_nodes()
            features[case_id]["edges"] = G.number_of_edges()
            features[case_id]["conn_comp"] = len(list(nx.connected_components(G)))

            edge_count = 0
            for x, y in points:
                for dx, dy in directions:
                    if (x + dx, y + dy) not in points:
                        edge_count += 1
                        break
            features[case_id]["edge_len"] = edge_count

        return features

    def parse_all_maps(self):
        """解析所有地图信息，统一格式"""
        cases_list = []
        features = {}
        map_files = [file_name for file_name in os.listdir(self.open_dir) if file_name.endswith('.map')]
        for file_name in tqdm(map_files, desc="解析地图信息", unit="字典"):
            file_path = os.path.join(self.open_dir, file_name)
            try:
                map_info = self.csr_matrix_dic(file_path)
                case_id = file_name.split('.')[0]
                features[case_id] = {}
                cases_list.append({
                    "case_id": case_id,
                    "map_data": map_info
                })
            except Exception as e:
                logging.error(f"加载地图 {file_name} 失败: {str(e)}")

        return cases_list
       

    def csr_matrix_dic(self, file_path):
        """读取地图文件并解析为稀疏矩阵"""
        with open(file_path, 'r') as file:
            lines = file.readlines()

        file_type = None
        height = None
        width = None
        crow = [0]
        col = []
        values = []
        non_zero_count = 0
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
                for j, char in enumerate(line):
                    if char != '@':
                        col.append(j)
                        values.append(1)
                        non_zero_count += 1
                crow.append(crow[-1] + non_zero_count)
                non_zero_count = 0

        crow = bm.tensor(crow, dtype=bm.int64)
        col = bm.tensor(col, dtype=bm.int64)
        values = bm.tensor(values, dtype=bm.float32)
        sparse_map = CSRTensor(crow,col,values)
        coords = [(i, c) for i, (start, end) in enumerate(zip(crow[:-1], crow[1:])) for c in col[start:end]]
        coords = bm.tensor(coords)

        result = {
            'height': height,
            'width': width,
            'maps': sparse_map,
            'coords':coords,
            'start': coords[0],
            'goal': coords[-1]
        }

        return result
    

def process_and_save_map(map_func, save_dir):
    """将二维0,1数组转换为. T @形式的数据并保存"""
    original_map = map_func()
    target_height = original_map.shape[0]
    target_width = original_map.shape[1]
    target_map = [['' for _ in range(target_width)] for _ in range(target_height)]

    for y in range(target_height):  # 进行 0 到 T、1 到 @ 的转换
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



def find_optimal_tables(input_file='data.json', output_file='optimal_tables.csv'):
    with open(input_file, 'r') as f:
        data = json.load(f)

    csv_rows = []
    for grid_key in data:
        grid = data[grid_key]
        
        features = {
            'obs_density': grid['obs_density'],
            'manhattan_dist': grid['manhattan_dist'],
            'nodes': grid['nodes'],
            'edges': grid['edges'],
            'conn_comp': grid['conn_comp'],
            'edge_len': grid['edge_len']
        }

        target = grid['target']
        best_algo = None
        min_composite = float('inf')
        for algo_name, algo_info in target.items():
            if algo_info['composite'] < min_composite:
                min_composite = algo_info['composite']
                best_algo = {
                    'algorithm': algo_name,
                    'params': algo_info['params']
                }

        csv_rows.append({**features, **best_algo})

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            'obs_density', 'manhattan_dist', 'nodes',
            'edges', 'conn_comp', 'edge_len',
            'algorithm', 'params'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"标签对数据已保存到 {output_file}")


if __name__ == "__main__":
    # 提取特征信息并保存
    open_dir = 'Data/test'
    preprocessor = MapDataPreprocessor(open_dir)
    # a = preprocessor.csr_matrix_dic('Data/test/ht_bartrand_n.map')
    # features = preprocessor.extract_features()
    # second_layer_dict = list(features.values())[0]
    # print(a)
    preprocessor.save_to_json()


    '''
    # 将0，1二维矩阵转换为“.”“T”“@”文件
    def sample_map_func():
        return bm.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    save_dir = 'Data/test'
    process_and_save_map(sample_map_func, save_dir)
    

    # 标签数据预处理
    find_optimal_tables(input_file="Data/test/Tables/Evaluate.json", output_file="Data/test/Tables/optimal_tables.csv")
    '''
