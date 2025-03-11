import gc
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from filelock import FileLock
import pandas as pd
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from fealpy.backend import backend_manager as bm
from datapre import MapDataPreprocessor
from main import ShortestPathApp

bm.set_backend('pytorch')
#device = 'cuda'
#bm.set_default_device(device)


parameter_spaces = {
    # bio_based
    "iwo_params": {
        "NP": (20, 500),            # 种群规模（整数） 
        "MaxIT": (100, 5000),       # 最大迭代次数（整数）
        "Nmax": (50, 500),          # 种群的最大数量（整数）
        "Smin": (0, 1),             # 每株杂草产生种子的最小数量（整数）
        "Smax": (5, 20),            # 每株杂草产生种子的最大数量（整数）
        "n": (2.0, 5.0),            # 控制扩散范围随迭代次数变化的指数参数
        "sigma_initial": (1.0, 5.0),# 扩散范围的初始值
        "sigma_final":(0.001, 0.1), # 扩散范围的最终值
    },
    # "mgo_params": {
    #     "NP": (20, 500),
    #     "MaxIT": (100, 5000), 
    #     "w": (1, 10.0),             # 步长系数
    #     "d1": (0, 1.0),             # 种群更新策略的阈值
    #     "rec_num": (5, 20),         # 记忆库的容量（整数）
    # },
    "prgbo_params": {
        "NP": (20, 500),
        "MaxIT": (100, 5000), 
    },
    # evolutionary_based
    "de_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),       
        "f": (0, 2.0),              # 交叉概率
        "cr": (0, 1.0),             # 变异因子          
    },
    "dcs_params":{
        "NP": (20, 500),       
        "MaxIT": (100, 5000),  
        "pc": (0, 1.0),             # 多样化操作的概率阈值
    },
    "ga_params":{
        "NP": (20, 500),       
        "MaxIT": (100, 5000),  
        "pc": (0, 1.0),             # 交叉概率
        "pm": (0, 0.1),             # 变异概率
    },
    # human_based
    "tlbo_params":{
        "NP": (20, 500),       
        "MaxIT": (100, 5000),              
    },
    # improved
    "iwoa_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),      
    },
    "cqpso_params":{
        "NP": (20, 500),       
        "MaxIT": (100, 5000),   
    },
    "dssa_params":{
        "NP": (20, 500),       
        "MaxIT": (100, 5000),   
        "Gc": (1.0, 2.0),           # 增益系数
        "Pdp": (0, 1.0),            # 概率阈值
        "dg": (0, 1.0),             # 一个系数
        "Cr": (0, 1.0),             # 交叉概率
    },
    "lqboa_params":{
        "NP": (20, 500),       
        "MaxIT": (100, 5000),   
    },
    "lqpso_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),   
        "sigma": (0.000001, 0.01),  # 早熟预防机制的阈值
        "delta": (0, 1.0),          # 粒子数量比例
    },
    # math_based
    "etoa_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),         
    },
    "sca_params":{
        "NP": (20, 500),       
        "MaxIT": (100, 5000),  
        "a": (1.0, 3.0),            # 控制参数 
    },
    # music_based
    "hsa_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000), 
        "HMCR": (0, 1.0),           # 和声记忆库考虑率
        "PAR": (0, 1.0),            # 音调调整率
        "FW_damp": (0.95, 0.999),   # 微调带宽阻尼系数
    },
    # physics_based
    "roa_params":{
        "NP": (20, 500),       
        "MaxIT": (100, 5000), 
        "w": (1.0, 10.0),           # 调整系数
    },
    "sao_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),         
    },
    # swarm_based
    "aco_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),   
        "alpha": (0, 5.0),          # 信息素重要程度因子
        "beta": (0, 10.0),          # 启发式信息重要程度因子
        "rho": (0, 1.0),            # 信息素挥发系数
        "Q": (0.1, 10.0),           # 信息素增加强度系数
    },
    "boa_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),  
        "a": (0, 1.0),               # 花香强度指数参数
        "c": (0.001, 0.1),           # 香气强度系数
        "p": (0, 1.0),               # 概率阈值        
    },
    "aro_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000), 
    },
    "bka_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000), 
        "p": (0, 1.0),             # 攻击行为和迁移行为的概率阈值        
    },
    "coa_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),    
    },
    "cpo_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),    
        "N_min": (30, 120),           # 动态种群数量的最小值（取整）
        "T": (2, 10),                 # 与迭代次数相关的周期参数（取整）
        "alpha": (0, 1.0),            # 加权系数
        "Tf": (0, 1.0),               # 搜索过程的概率阈值
    },
    "cdw_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),   
        "p": (0, 1.0),                 # 概率阈值
    },
    "cso_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),  
        "alpha": (0.001, 1.0),           # 步长控制参数
        "p": (0, 1.0),                  # 发现概率控制参数          
    },
    "gwo_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),         
    },
    "hho_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),   
    },
    "hoa_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),   
    },
    "hba_params": { 
        "NP": (20, 500),       
        "MaxIT": (100, 5000),    
        "c": (1.0, 5.0),                 # 收敛速率常数因子
        "beta": (3.0, 10.0),             # 搜索行为常数因子
    },
    "jso_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),    
    },
    "mpa_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),    
        "p": (0.1, 1.0),                # 步长的系数
        "FADs": (0, 1.0),               # 聚集装置影响概率
    },
    "pso_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),  
        "c1": (1.0, 4.0),               # 个体学习因子
        "c2": (1.0, 4.0),               # 群体学习因子
        "w_max": (0.8, 1.0),            # 最大惯性权重 
        "w_min":(0.2, 0.5),             # 最小惯性权重
    },
    "qpso_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),   
        "alpha_max": (0.8, 1.0),        # 收缩 - 扩张系数的最大值
        "alpha_min": (0.2, 0.5),        # 收缩 - 扩张系数的最小值     
    },
    "scso_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),      
    },
    "soa_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),  
        "Fc": (1.0, 5.0),              # 更新作用的系数
        "u": (0.5, 2.0),               # 指数项的系数
        "v": (0.5, 2.0),               # 指数项的系数        
    },
    "ssam_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),  
        "ST": (0, 1.0),               # 预警值
        "PD": (0, 1.0),               # 发现者比例
        "SD": (0, 1.0),               # 警戒者比例        
    },
    "ssas_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),  
        "Gc": (1.0, 3.0),              # 引力常数
        "Pdp": (0, 1.0),               # 概率值 
    },
    "sfoa_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),  
    },
    "woa_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),  
    },
    "zoa_params": {
        "NP": (20, 500),       
        "MaxIT": (100, 5000),  
    },
}


"""性能评估"""
class PerformanceEvaluator:
    def __init__(self, test_cases, output_dir="Data", n_trials=20):   
        self.COLUMN_ORDER = ['algo', 'params', 'composite',
                             'fitness', 'turns', 'time']   
        self.output_dir = Path(output_dir) / 'Tables'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parameter_spaces = parameter_spaces 
        self.test_cases = test_cases  
        self.n_trials = n_trials                                     


    def evaluate_all(self, algorithms=None):
        """执行完整评估流程"""
        algorithms = algorithms or self.parameter_spaces.keys()
        
        tasks = []
        for case in self.test_cases:
            for algo in algorithms:
                tasks.append((case, algo)) 
                    
        max_workers = 4  # 手动设置最大工作进程数     
        with ProcessPoolExecutor(max_workers=max_workers) as executor:                             # 使用并行加速
            futures = [
                executor.submit(self._evaluate_single_task, case, algo)
                for case, algo in tasks
            ]
            with tqdm(total=len(futures), desc="全局进度") as pbar:        
                for future in futures:
                    future.result()
                    pbar.update(1)


    def _evaluate_single_task(self, case, algo):
        """单个评估任务（地图+算法）"""
        task_file_name = "Evaluate.json"
        output_path = self.output_dir / task_file_name
        lock_path = output_path.with_suffix(".lock")

        param_combos, _ = self._generate_parameters(algo, case)  # 生成参数组合
        
        results = []
        case_id = case['case_id']
        for params in param_combos:
            try:
                record = self._evaluate_single_run(algo, params, case)
                results.append(record)   # 评估所有参数组合
            except Exception as e:
                logging.error(f"评估失败 {case_id}-{algo}: {str(e)}")

        if results:
            df_results = pd.DataFrame(results)
            best_result = df_results.loc[df_results['composite'].idxmin()].to_dict()  # 获取当前算法最优结果
            print(best_result)

            with FileLock(lock_path): # 使用文件锁安全写入
                with open(output_path, 'r') as json_file:
                    existing_features = json.load(json_file)

                if case_id in existing_features and "target" in existing_features[case_id] and algo in existing_features[case_id]["target"]:
                    existing_result  = existing_features[case_id]["target"][algo]
                    if "composite" in existing_result:
                        existing_composite = existing_result["composite"]
                        new_composite = best_result["composite"]
                        if new_composite < existing_composite:
                            existing_features[case_id]["target"][algo] = best_result
                    else:
                        existing_features[case_id]["target"][algo] = best_result
                
                with open(output_path, 'w') as json_file:
                    print(existing_features)
                    json.dump(existing_features, json_file, indent=4)


    def _generate_parameters(self, algo_name, case):
        """生成算法的参数组合"""
        space = []
        dim_names = []   
        param_space = self.parameter_spaces[f"{algo_name.lower()}_params"]
        if not param_space:
            raise ValueError(f"参数空间未定义: {algo_name}")   

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
            opt = result['composite']
            return opt

        # 运行贝叶斯优化
        res = gp_minimize(
            objective, 
            space, 
            n_calls=self.n_trials,
            acq_func='EI',          # 改用期望改进采集函数
            noise=1e-5,             # 添加微小噪声处理评分相同情况
            random_state=0
            )

        param_combo_dic = []
        for x_iter in res.x_iters:
            param_dict = dict(zip(dim_names, x_iter))
            param_combo_dic.append(param_dict)

        return param_combo_dic, res.func_vals
        

    def _evaluate_single_run(self, algo_name, params, case):
        """单次参数组合评估"""
        try:
            if not hasattr(self, '_app_cache'):   # 使用缓存避免重复初始化
                self._app_cache = {}

            cache_key = (algo_name, case['case_id'])
            special_params = {key: value for key, value in params.items() if key not in ['NP', 'MaxIT']}
            if cache_key not in self._app_cache: # 如果缓存中没有当前算法的应用实例，则创建新的实例
                self._app_cache[cache_key] = ShortestPathApp(
                    case['map_data'], 
                    case['map_data']['start'],  # 可以自行取值
                    case['map_data']['goal'], 
                    algo_chosen=[algo_name],
                    NP=params.get('NP', 50),
                    MaxIters=params.get('MaxIT', 300),
                    **special_params
                )
            app = self._app_cache[cache_key]
            
            results = app.optimize() 
            run_time = results[algo_name]['time']

            record = self._build_record(algo_name, params, results, run_time)

            return record
        
        except Exception as e:
            logging.error(f"评估失败: {algo_name} on {case['case_id']} - {str(e)}")
            return bm.inf 
        
        finally: # 显式释放内存
            if 'app' in locals():
                del app  # 删除当前的 app 实例
                self._app_cache.pop(cache_key, None)  # 从缓存中移除相应的 cache_key
            gc.collect()  # 手动触发垃圾回收
        


    def _build_record(self, algo_name, params, results, run_time):
        """统一构建结果记录"""       
        priority_weights = {'fitness': 1e4, 'turns': 1e3, 'time': 1}        # 复合评分计算
        composite = (
            results[algo_name]['gbest_f'] * priority_weights['fitness'] +
            results[algo_name]['truns'] * priority_weights['turns'] +
            run_time * priority_weights['time']
            )
        
        return {  
            # 'params': json.dumps(params, cls=NumpyEncoder),                 # 自定义编码器
            'params': params,
            'composite': composite.item(),                                 
            'fitness': float(results[algo_name]['gbest_f']),
            'turns': int(results[algo_name]['truns']),
            'time': run_time
            }

if __name__ == "__main__":
    
    file_path = 'Data/test'
    preprocessor = MapDataPreprocessor(file_path)
    test_cases = preprocessor.parse_all_maps()
    # algorithms_to_test = ["DE", "SAO", "BOA", "BKA", "CSO", "GWO", "HBA"]
    algorithms_to_test = ["DE"]
    # print(test_cases)
           
    evaluator = PerformanceEvaluator(test_cases, output_dir = file_path, n_trials=10)
    evaluator.evaluate_all(algorithms=algorithms_to_test)
