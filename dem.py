import os
import json
import time
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from filelock import FileLock
import networkx as nx
import pandas as pd
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from fealpy.backend import backend_manager as bm
from tabulate import tabulate
import multiprocessing


from AGV_maps import short_folder
from main import ShortestPathApp


import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """处理numpy数据类型的JSON编码"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def parse_maps_in_folder(folder_path):
    """解析文件夹中的所有地图文件，并返回统一格式的测试用例"""
    test_cases = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.map'):
            file_path = os.path.join(folder_path, file_name)
            try:
                map_info = short_folder(file_path)
                test_cases.append({
                    "case_id": file_name.split('.')[0],
                    "map_data": map_info,
                    "start": map_info['start'],
                    "goal": map_info['goal']
                })
            except Exception as e:
                logging.error(f"加载地图 {file_name} 失败: {str(e)}")
    return test_cases


def extract_map_features(map_data, start, goal):
    """提取路径规划问题的地图特征"""
    features = {}
    sparse_map = map_data['maps']
    height, width = map_data['height'], map_data['width']

    rows, cols = sparse_map.nonzero()  # 可行点坐标列表
    points = list(zip(rows, cols))

    total_pixels = height * width
    free_pixels = len(rows)
    features["obs_density"] = 1 - (free_pixels / total_pixels)  # 障碍物密度
    features["manhattan_dist"] = abs(start[0] - goal[0]) + abs(start[1] - goal[1])  # 起点终点曼哈顿距离

    G = nx.Graph()
    G.add_nodes_from(points)  # 添加可行点作为节点
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右/下/左/上
    for x, y in points:
        for dx, dy in directions:
            nx_, ny = x + dx, y + dy
            if (nx_, ny) in points:
                G.add_edge((x, y), (nx_, ny))
    features["nodes"] = G.number_of_nodes()
    features["edges"] = G.number_of_edges()
    features["conn_comp"] = len(list(nx.connected_components(G)))

    edge_count = 0
    for x, y in points:
        for dx, dy in directions:
            if (x + dx, y + dy) not in points:
                edge_count += 1
                break
    features["edge_len"] = edge_count
    return features


parameter_spaces = {
    "sao_params": {
        "np_size": (20, 500),  # 种群规模（整数）
        "max_iter": (100, 1000),  # 最大迭代次数（整数）
    },
    "coa_params": {
        "np_size": (20, 500),
        "max_iter": (100, 1000),
    },
    "hba_params": {
        "np_size": (20, 500),
        "max_iter": (100, 1000),
        "C": (1.0, 3.0),  # 收敛速率常数因子
        "beta": (5.0, 10.0),  # 搜索行为常数因子
    },
    "qpso_params": {
        "np_size": (20, 500),
        "max_iter": (100, 1000),
        "alpha_max": (0.8, 1.0),  # 收缩-扩张系数
        "alpha_min": (0.1, 0.5),
    },
    "pso_params": {
        "np_size": (20, 500),
        "max_iter": (100, 1000),
        "c1": (1.5, 2.5),  # 个体认知常数
        "c2": (1.5, 2.5),  # 社会认知常数
        "w_max": (0.9, 1.0),  # 惯性权重
        "w_min": (0.4, 0.5),
    },
    "gwo_params": {
        "np_size": (20, 500),
        "max_iter": (100, 1000),
    },
    "cpo_params": {
        "np_size": (20, 1000),
        "max_iter": (100, 1000),
        "N_min": (30, 120),  # 最小种群数
        "T": (2, 10),  # 种群规模变化的速率
        "alpha": (0, 1.0),  # 加权系数
        "Tf": (0.7, 0.9),  # 搜索过程的“阈值”
    },
    "bka_params": {
        "np_size": (30, 500),
        "max_iter": (100, 1000),
        "p": (0.6, 0.9),  # 攻击行为和迁移行为的比例
    },
    "boa_params": {
        "np_size": (20, 500),
        "max_iter": (100, 1000),
        "a": (0, 2.0),  # 香气影响的强度
        "c": (0, 1.0),  # 香气强度系数
        "p": (0.5, 0.9),  # 全局搜索和局部搜索的概率
    },
    "cs_params": {
        "np_size": (20, 500),
        "max_iter": (100, 1000),
        "alpha": (0.01, 1.0),  # 步长控制参数
        "Pa": (0.1, 0.4),  # 发现概率控制参数
    },
    "de_params": {
        "np_size": (20, 500),
        "max_iter": (100, 1000),
        "F": (0.1, 1.0),  # 交叉概率
    },
    "eto_params": {
        "np_size": (20, 500),
        "max_iter": (100, 2000),
    }
}


"""性能评估"""


class PerformanceEvaluator:
    COLUMN_ORDER = [
        'map_id', 'obs_density', 'manhattan_dist', 'nodes',
        'edges', 'conn_comp', 'edge_len', 'algo', 'params',
        'fitness', 'turns', 'time', 'composite'
    ]

    def __init__(self, test_cases, output_dir="results"):
        self.test_cases = self._preprocess_cases(test_cases)  # 地图特征预处理
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.parameter_spaces = parameter_spaces  # 加载参数空间定义
        self.test_cases = test_cases  # 标准测试集（包含多个地图数据）

    def _preprocess_cases(self, raw_cases):
        """预处理并缓存特征"""
        processed = []
        for case in raw_cases:
            features = extract_map_features(case["map_data"], case["start"], case["goal"])
            case['features'] = {
                k: float(v) if isinstance(v, float) else int(v)
                for k, v in features.items()
            }
            processed.append(case)
        return processed

    def evaluate_all(self, algorithms=None, n_trials=30):
        """执行完整评估流程"""
        algorithms = algorithms or self.parameter_spaces.keys()
        tasks = [(case, algo) for case in self.test_cases for algo in algorithms]  # 创建任务列表
        total_progress = len(tasks) * n_trials  # 计算总进度
        progress_queue = multiprocessing.Queue()

        with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as executor:  # 使用并行加速
            futures = [
                executor.submit(self._evaluate_single_task, case, algo, n_trials, progress_queue)
                for case, algo in tasks
            ]

            with tqdm(total=total_progress, desc="全局进度") as pbar:
                while any([not future.done() for future in futures]) or not progress_queue.empty():
                    if not progress_queue.empty():
                        try:
                            pbar.update(progress_queue.get(timeout=0.1))
                        except multiprocessing.queues.Empty:
                            pass

        self.merge_all_results()

    def _evaluate_single_task(self, case, algo, n_trials, progress_queue):
        """单个评估任务（地图+算法）"""
        case_id = case['case_id']
        task_file_name = f"{case_id}_{algo}.csv"
        output_path = self.output_dir / task_file_name
        lock_path = output_path.with_suffix(".lock")

        param_combos, _ = self._generate_parameters(algo, case, n_trials)  # 生成参数组合

        algo_results = []
        for params in param_combos:
            try:
                record = self._evaluate_single_run(algo, params, case)
                algo_results.append(record)  # 评估所有参数组合
                progress_queue.put(1)
                print(f"完成参数评估: 地图ID={case_id}, 算法={algo}, 参数={params}")
            except Exception as e:
                logging.error(f"评估失败 {case_id}-{algo}: {str(e)}")

        if algo_results:
            df_algo = pd.DataFrame(algo_results)
            best_record = df_algo.loc[df_algo['composite'].idxmin()].to_dict()  # 获取当前算法最优结果

            with FileLock(lock_path):  # 使用文件锁安全写入
                df_best = pd.DataFrame([best_record])
                df_best[self.COLUMN_ORDER].to_csv(output_path, index=False)

    def _generate_parameters(self, algo_name, case, n_calls=50):
        """生成算法的参数组合"""
        space = []
        dim_names = []
        param_space = self.parameter_spaces[f"{algo_name.lower()}_params"]
        if not param_space:
            raise ValueError(f"参数空间未定义: {algo_name}")

        # 构建优化空间
        for param, bounds in param_space.items():
            if isinstance(bounds, tuple):
                if all(isinstance(b, int) for b in bounds):
                    dim = Integer(*bounds, name=param)
                else:
                    dim = Real(*bounds, name=param)
                space.append(dim)
                dim_names.append(param)
            elif isinstance(bounds, list):
                space.append(Categorical(bounds, name=param))
                dim_names.append(param)

        # 定义目标函数
        @use_named_args(space)
        def objective(**params):
            result = self._evaluate_single_run(algo_name, params, case)
            return result['composite']

        # 运行贝叶斯优化

        res = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            acq_func='EI',  # 改用期望改进采集函数
            noise=1e-5,  # 添加微小噪声处理评分相同情况
            random_state=0
        )

        # 将列表转换为字典
        param_combo_dic = []
        for x_iter in res.x_iters:
            param_dict = dict(zip(dim_names, x_iter))
            param_combo_dic.append(param_dict)

        return param_combo_dic, res.func_vals

    def _evaluate_single_run(self, algo_name, params, case):
        """单次参数组合评估"""
        try:
            if not hasattr(self, '_app_cache'):  # 使用缓存避免重复初始化
                self._app_cache = {}

            cache_key = (algo_name, case['case_id'])

            if cache_key not in self._app_cache:  # 如果缓存中没有当前算法的应用实例，则创建新的实例
                self._app_cache[cache_key] = ShortestPathApp(
                    case['map_data'],
                    case['start'],
                    case['goal'],
                    algo_chosen=[algo_name],
                    NP=params.get('np_size', 100),
                    MaxIters=params.get('max_iter', 300)
                )
            app = self._app_cache[cache_key]

            results = app.optimize()
            run_time = results[algo_name]['time']

            record = self._build_record(algo_name, params, case, results, run_time)
            # 添加输出代码
            print('_________')
            print(f"地图ID: {case['case_id']}, 算法: {algo_name}, 参数: {params}")
            print(f"适应度: {record['fitness']}, 转弯次数: {record['turns']}, 运行时间: {record['time']}, 综合评分: {record['composite']}")

            return record

        except Exception as e:
            logging.error(f"评估失败: {algo_name} on {case['case_id']} - {str(e)}")
            return bm.inf

        finally:  # 显式释放内存
            if 'app' in locals():
                del app  # 删除当前的 app 实例
                self._app_cache.pop(cache_key, None)  # 从缓存中移除相应的 cache_key

    def _build_record(self, algo_name, params, case, results, run_time):
        """统一构建结果记录"""
        priority_weights = {'fitness': 1e4, 'turns': 1e3, 'time': 1}  # 复合评分计算
        composite = (
                results[algo_name]['gbest_f'] * priority_weights['fitness'] +
                results[algo_name]['truns'] * priority_weights['turns'] +
                run_time * priority_weights['time']
        )
        return {
            'map_id': case['case_id'],
            **case['features'],  # 直接使用预处理特征
            'algo': algo_name,
            'params': json.dumps(params, cls=NumpyEncoder),  # 自定义编码器
            'fitness': float(results[algo_name]['gbest_f']),
            'turns': int(results[algo_name]['truns']),
            'time': run_time,
            'composite': composite
        }
    def merge_all_results(self):
        """合并所有任务的结果文件"""
        result_files = list(self.output_dir.glob("*.csv"))
        if not result_files:
            raise ValueError("未找到结果文件")

        df = pd.concat([pd.read_csv(f) for f in result_files])
        final_output_path = self.output_dir / "all_results.csv"
        df[self.COLUMN_ORDER].to_csv(final_output_path, index=False)
        print(f"所有任务结果已合并到 {final_output_path}")   

    def analyze_results(self):
        """增强结果分析"""
        # result_files = list(self.output_dir.glob("*.csv"))
        result_file = self.output_dir / "all_results.csv"
        # if not result_files:
        if not result_file.exists():
            raise ValueError("未找到结果文件")
        
        # df = pd.concat([pd.read_csv(f) for f in result_files])
        df = pd.read_csv(result_file)

        analysis = (
            df.groupby(['map_id', 'algo'])
            .agg({
                'fitness': 'min',
                'turns': 'min',
                'time': 'min',
                'composite': 'min'
            })
            .reset_index()
            .sort_values(['map_id', 'composite'])
        )

        report  = []
        for (map_id), group in analysis.groupby('map_id'):
            best = group.nsmallest(1, 'composite').iloc[0]
            report.append({
                "地图ID": map_id,
                "最佳算法": best['algo'],
                "综合评分": f"{best['composite']:.1f}",
                "路径长度": f"{best['fitness']:.2f}",
                "耗时(s)": f"{group['time'].min():.3f}",
                "候选算法数": len(group) - 1
            })
    
        print("\n" + "="*80)
        print("最终分析报告".center(80))
        print("="*80)
        print(tabulate(report, headers="keys", tablefmt="grid", stralign="center"))
        return analysis  


if __name__ == "__main__":
    
    file_path = 'Data/test'
    test_cases = parse_maps_in_folder(file_path)

    algorithms_to_test = ["SAO", "DE"]

    # 执行评估（每个算法在每个测试用例上运行 n_trials 次参数组合）            
    evaluator = PerformanceEvaluator(test_cases, output_dir="results_v2")
    evaluator.evaluate_all(algorithms=algorithms_to_test, n_trials=10)
    # evaluator.analyze_results()
