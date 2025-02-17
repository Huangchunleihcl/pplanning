import time
import logging
from fealpy.backend import backend_manager as bm
from fealpy.opt import *
from fealpy.opt.optimizer_base import opt_alg_options
# from AGV_maps import MAP_data as MAPdata
from AGVs_maps import AGV_data as AGVdata
from CVRP_maps import CVRP_data as CVRPdata
from TSP_citys import TSProblem
from AGV_maps import AGVProblem
from AGVs_maps import AGVsProblem
from CVRP_maps import CVRProblem
from TSP_citys import tsp_folder, print_results_tsp, printroute_tsp
from AGV_maps import short_folder, print_results_short, printroute_short
from AGVs_maps import print_results_ob, printroute_ob
from CVRP_maps import print_results_cvrp, printroute_cvrp

# bm.set_backend('pytorch')
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TSPOptimizerApp:
    def __init__(self, map_chosen = None, algo_chosen = None, NP=200, lb=0, ub=1, MaxIters = 10000):
        self.NP = NP
        self.lb = lb
        self.ub = ub
        self.MaxIters = MaxIters

        self.algo_chosen = ['SAO'] if algo_chosen is None else algo_chosen
        self.num_algo = len(self.algo_chosen)
        self.citys = map_chosen['citys']
        self.dim = map_chosen['dimension']

        self.TSPtest = TSProblem(self.citys)
        self.TSPtest.calD()
        self.D = self.TSPtest.D
        self.fobj = lambda x: self.TSPtest.fitness(x)

        self.optimizers = {
            'SAO': SnowAblationOpt,
            'COA': CrayfishOptAlg,
            'HBA': HoneybadgerAlg,
            'QPSO': QuantumParticleSwarmOpt,
            'PSO': ParticleSwarmOpt,
            'GWO': GreyWolfOpt,
            'ACO': AntColonyOptAlg,
            'HO': HippopotamusOptAlg,
            'CPO': CrestedPorcupineOpt,
            'BKA':BlackwingedKiteAlg,
            'BOA':ButterflyOptAlg,
            'CS':CuckooSearchOpt,
            'DE':DifferentialEvolution,
            'ETO':ExponentialTrigonometricOptAlg,
        }
        self.results = {}

    def optimize(self):
        for algo_idx in range(self.num_algo):
            algo_name  = self.algo_chosen[algo_idx]
            algo_optimizer = self.optimizers[algo_name]

            x0 = self.lb + bm.random.rand(self.NP, self.dim) * (self.ub - self.lb)
            option = opt_alg_options(x0, self.fobj, (self.lb, self.ub), self.NP, self.MaxIters) 

            algo_start = time.perf_counter()
            if algo_name == 'ACO':
                optimizer = algo_optimizer(option, self.D)              
            else:
                optimizer = algo_optimizer(option)
            
            # gbest, gbest_f = optimizer.run() 
            optimizer.run()
            gbest = optimizer.gbest
            gbest_f = optimizer.gbest_f        
            gbest = bm.argsort(gbest)
            gbest_f = float(gbest_f)

            route = bm.concatenate((gbest, gbest[0: 1]))
            route_citys = self.citys[route]    

            algo_end = time.perf_counter()
            algo_runtime = algo_end - algo_start

            self.results[algo_name] = {
                'gbest_f': gbest_f, 
                'route': route,
                'time': algo_runtime,
                'route_citys': route_citys,  
                }

        # print_results_tsp(self.results)
        # printroute_tsp(self.citys, self.results)
        return self.results


class ShortestPathApp:
    def __init__(self, maps, start = None, goal = None, algo_chosen = None, NP=130, lb=0, ub=1, MaxIters = 30):
        self.NP = NP
        self.lb = lb
        self.ub = ub
        self.MaxIters = MaxIters
        self.maps = maps['maps'] # 稀疏矩阵
        self.start = maps['start'] if start is None else start
        self.goal = maps['goal'] if goal is None else goal
        self.algo_chosen = ['SAO'] if algo_chosen is None else algo_chosen

        if self.maps[self.start[0], self.start[1]] != 1 or self.maps[self.goal[0], self.goal[1]] != 1: 
            raise ValueError("Error: Invalid start or goal point")
        
         # 初始化路径规划问题
        self.Shorttest = AGVProblem(self.maps, self.start, self.goal)
        self.Shorttest.calD()
        self.D = self.Shorttest.data['D']               # 距离矩阵
        self.dim = self.D.shape[0]                      # 可行点的数量
        self.fobj = lambda x: self.Shorttest.fitness(x)        
        
        # 优化算法映射
        self.optimizers = {
            'SAO': SnowAblationOpt,
            'COA': CrayfishOptAlg,
            'HBA': HoneybadgerAlg,
            'QPSO': QuantumParticleSwarmOpt,
            'PSO': ParticleSwarmOpt,
            'GWO': GreyWolfOpt,
            'ACO': AntColonyOptAlg,
            'HO': HippopotamusOptAlg,
            'CPO': CrestedPorcupineOpt,
            'BKA':BlackwingedKiteAlg,
            'BOA':ButterflyOptAlg,
            'CS':CuckooSearchOpt,
            'DE':DifferentialEvolution,
            'ETO':ExponentialTrigonometricOptAlg,
        }
        self.results = {}

    def optimize(self):
        x0 = self.lb + bm.random.rand(self.NP, self.dim) * (self.ub - self.lb)
        option = opt_alg_options(x0, self.fobj, (self.lb, self.ub), self.NP, self.MaxIters)          
        
        # 运行优化算法
        for algo_name in self.algo_chosen:
            # logging.info(f"Running {algo_name}...")
            algo_optimizer = self.optimizers[algo_name]

            algo_start = time.perf_counter()

            # 初始化优化器
            if algo_name == 'ACO':
                optimizer = algo_optimizer(option, self.D)
            else:
                optimizer = algo_optimizer(option)
            optimizer.run()
            gbest = optimizer.gbest
            gbest_f = optimizer.gbest_f 

            algo_end  = time.perf_counter()
            algo_runtime  = algo_end - algo_start

            x0[0] = gbest
            route = self.Shorttest.gbestroute(x0)
            gbest_f = float(gbest_f - self.Shorttest.data['trun_weight'] * route['truns'])
            
            self.results[algo_name] = {
                "gbest_f": gbest_f,
                "truns":route['truns'], 
                "route": route['path'],
                "time": algo_runtime,
                }

        return self.results


class ObstacleAvoidanceApp:
    def __init__(self, map_sets = None, starts = None, goals = None, algo_chosen = None, NP=120, lb=0, ub=1, MaxIters = 50):
        self.NP = NP
        self.lb = lb
        self.ub = ub
        self.MaxIters = MaxIters  

        self.map_sets = 0 if map_sets is None else map_sets
        self.maps = AGVdata[self.map_sets]['map']
        self.starts = AGVdata[self.map_sets]['start'] if starts is None else starts
        self.goals = AGVdata[self.map_sets]['goal'] if goals is None else goals
        self.algo_chosen = ['SAO'] if algo_chosen is None else algo_chosen

        self.num_agvs = len(self.starts)
        self.num_algo = len(self.algo_chosen)
        self.dim = (self.maps == 0).sum()

        self.OAtest = AGVsProblem(self.maps, self.starts, self.goals, self.num_agvs, self.dim)
        self.OAtest.calD()
        self.D = self.OAtest.data['D']
        self.fobj = lambda x: self.OAtest.fitness(x)        
        
        self.optimizers = {
            'SAO': SnowAblationOpt,
            'COA': CrayfishOptAlg,
            'HBA': HoneybadgerAlg,
            'QPSO': QuantumParticleSwarmOpt,
            'PSO': ParticleSwarmOpt,
            'GWO': GreyWolfOpt,
            'ACO': AntColonyOptAlg,
            'HO': HippopotamusOptAlg,
            'CPO': CrestedPorcupineOpt,
            'BKA':BlackwingedKiteAlg,
            'BOA':ButterflyOptAlg,
            'CS':CuckooSearchOpt,
            'DE':DifferentialEvolution,
            'ETO':ExponentialTrigonometricOptAlg,
        }
        self.results = {}

    def optimize(self):
        for agv in range(self.num_agvs):
            if self.maps[self.starts[agv][0]][self.starts[agv][1]] != 0 or self.maps[self.goals[agv][0]][self.goals[agv][1]] != 0: 
                print("Error: Wrong start point or goal point")
                return
        
        x0 = self.lb + bm.random.rand(self.NP, self.dim) * (self.ub - self.lb)
        option = opt_alg_options(x0, self.fobj, (self.lb, self.ub), self.NP, self.MaxIters)          
        
        for algo_idx in range(self.num_algo):
            algo_name  = self.algo_chosen[algo_idx]
            algo_optimizer = self.optimizers[algo_name]

            algo_start = time.perf_counter()
            if algo_name == 'ACO':             
                optimizer = algo_optimizer(option, self.D)
            else:
                optimizer = algo_optimizer (option)
            gbest, _ = optimizer.run()
            gbest = bm.argsort(gbest)
            algo_end  = time.perf_counter()
            algo_runtime  = algo_end - algo_start
            
            route = self.OAtest.gbestroute(gbest)

            self.results[algo_name] = {
                "gbest_f": route['fit'],
                "time": algo_runtime,
                }
            for agv in range(self.num_agvs):
                if 'AGV' not in self.results[algo_name]:
                    self.results[algo_name]['AGV'] = {}
                self.results[algo_name]['AGV'][agv] = {
                    "fit_agv":len(route['path'][agv]) - 1,
                    "route": route['path'][agv],
                }

        print_results_ob(self.results)
        printroute_ob(self.maps, self.results)


class CVROptimizerApp:
    def __init__(self, map_sets = None, algo_chosen = None, NP=130, lb=0, ub=1, MaxIters = 10000):
        self.NP = NP
        self.lb = lb
        self.ub = ub
        self.MaxIters = MaxIters

        self.map_sets = 0 if map_sets is None else map_sets
        self.algo_chosen = ['SAO'] if algo_chosen is None else algo_chosen
        self.num_algo = len(self.algo_chosen)
        self.citys = CVRPdata[self.map_sets]['NAME'][:, :2] # 客户坐标
        self.demands = CVRPdata[self.map_sets]['NAME'][:, 2] # 客户需求量
        self.dim = CVRPdata[self.map_sets]['DIMENSION'] # 客户 + 仓库总数
        self.capacity = CVRPdata[self.map_sets]['CAPACITY']  # 获取车辆容量
        self.trucks = CVRPdata[self.map_sets]['No_of_trucks'] # 车辆总数

        self.CVRPtest = CVRProblem(self.citys, self.demands, self.dim, self.capacity, self.trucks)
        self.CVRPtest.calD()
        self.D = self.CVRPtest.D
        self.fobj = lambda x: self.CVRPtest.fitness(x)

        self.optimizers = {
            'SAO': SnowAblationOpt,
            'COA': CrayfishOptAlg,
            'HBA': HoneybadgerAlg,
            'QPSO': QuantumParticleSwarmOpt,
            'PSO': ParticleSwarmOpt,
            'GWO': GreyWolfOpt,
            'ACO': AntColonyOptAlg,
            'HO': HippopotamusOptAlg,
            'CPO': CrestedPorcupineOpt,
            'BKA':BlackwingedKiteAlg,
            'BOA':ButterflyOptAlg,
            'CS':CuckooSearchOpt,
            'DE':DifferentialEvolution,
            'ETO':ExponentialTrigonometricOptAlg,
        }
        self.results = {}

    def optimize(self):
        # start_time = time.perf_counter()
        for algo_idx in range(self.num_algo):
            algo_name  = self.algo_chosen[algo_idx]
            algo_optimizer = self.optimizers[algo_name]

            algo_start = time.perf_counter()
            if algo_name == 'ACO':
                NP = 50
                MaxIters = 100
                x0 = self.lb + bm.random.rand(NP, self.dim) * (self.ub - self.lb)
                option = opt_alg_options(x0, self.fobj, (self.lb, self.ub), NP = NP, MaxIters = MaxIters) 
                optimizer = algo_optimizer(option, self.D)
                # gbest, gbest_f = optimizer.run() 
                optimizer.run()
                gbest = optimizer.gbest
                gbest_f = optimizer.gbest_f  
            elif algo_name == 'HO':
                NP = 200
                MaxIters = 100
                x0 = self.lb + bm.random.rand(NP, self.dim) * (self.ub - self.lb)
                option = opt_alg_options(x0, self.fobj, (self.lb, self.ub), NP = NP, MaxIters = MaxIters) 
                optimizer = algo_optimizer(option)
                # gbest, gbest_f = optimizer.run()    
                optimizer.run()
                gbest = optimizer.gbest
                gbest_f = optimizer.gbest_f                          
            else:
                x0 = self.lb + bm.random.rand(self.NP, self.dim) * (self.ub - self.lb)
                option = opt_alg_options(x0, self.fobj, (self.lb, self.ub), self.NP, self.MaxIters) 
                optimizer = algo_optimizer(option)
                # gbest, gbest_f = optimizer.run()   
                optimizer.run()
                gbest = optimizer.gbest
                gbest_f = optimizer.gbest_f  
            gbest = bm.argsort(gbest)

            _, route = self.CVRPtest.gbestroute_cvrp(gbest)    
            route_citys = [[self.citys[idx].tolist() for idx in sub_route] for sub_route in route]        

            algo_end = time.perf_counter()
            algo_runtime = algo_end - algo_start

            self.results[algo_name] = {
                'gbest_f': gbest_f, 
                'route': route,
                'time': algo_runtime,
                'route_citys': route_citys,  
                }
        # end_time = time.perf_counter()
        # running_time = end_time - start_time
        print_results_cvrp(self.results)
        printroute_cvrp(self.citys, self.results)
        # print("Total runtime: ", running_time)
 
        return self.results


if __name__ == "__main__":
    
    # 不指定则 = None
    map_sets = 1

    # algo_chosen = ['SAO','COA','HBA','QPSO','PSO','GWO', 'CPO','BKA','BOA','CS','DE','ETO','HO']
    algo_chosen = ['SAO']

    # [94, 23, 0, 6, 2, 38, 7, 34, 36, 54]
    # Traveling Salesman Problem
    '''
    print('\n---------TSPOptimizer---------')
    folder_path  = "Data/TSPLIB"
    TSPdata = tsp_folder(folder_path)
    map_chosen = TSPdata[map_sets]
    print(map_chosen['name'])
    tsp_optimizer = TSPOptimizerApp(algo_chosen=algo_chosen, map_chosen=map_chosen)
    tsp_optimizer.optimize()
    '''


    # Shortest Path Problem
    # '''
    logging.info('\n---------ShortestPath---------')
    file_path = 'Data/CA_CAVEMAP/0_grid_2.map'
    maps = short_folder(file_path)
    start = None
    goal = None
    short_optimizer = ShortestPathApp(algo_chosen=algo_chosen, maps=maps, start=start, goal=goal)
    results = short_optimizer.optimize()
    
    print_results_short(results)
    printroute_short(maps['maps'], results)
    # '''


    # Obstacle Avoidance Problem
    '''
    print('\n---------Obstacle Avoidance---------')
    folder_path  = "Data/STREETMAP"
    starts = [[0, 0],[19,4],[5, 10]]   
    goals = [[19, 19],[1,19],[15,10]]
    oa_optimizer = ObstacleAvoidanceApp(algo_chosen=algo_chosen, map_sets=map_sets, starts=starts, goals=goals)
    oa_optimizer.optimize()
    '''

    # Capacitated Vehicle Routing Problem
    '''
    print('\n---------CVRPOptimizer---------')
    folder_path  = "Data/CVRPLIB"
    cvrp_optimizer = CVROptimizerApp(algo_chosen=algo_chosen, map_sets=map_sets)  
    cvrp_optimizer.optimize()   
    '''



