from fealpy.backend import backend_manager as bm
import matplotlib.pyplot as plt
import os
import csv
import shutil

def P_n21_k2():
    citys = bm.array([
        [30, 40, 0],
        [37, 52, 7],
        [49, 49, 30],
        [52, 64, 16],
        [31, 62, 23],
        [52, 33, 11],
        [42, 41, 19],
        [52, 41, 15],
        [57, 58, 28],
        [62, 42, 8],
        [42, 57, 8],
        [27, 68, 7],
        [43, 67, 14],
        [58, 48, 6],
        [58, 27, 19],
        [37, 69, 11],
        [38, 46, 12],
        [61, 33, 26],
        [62, 63, 17],
        [63, 69, 6],
        [45, 35, 15],
    ])
    return citys


CVRP_data = [
    {
        "NAME": P_n21_k2(),
        "DIMENSION": 21,
        "No_of_trucks": 2,
        "CAPACITY": 160,
        "DEPOT_SECTION": 1,
        "Optimalvalue": 211,
    }
]


class CVRProblem:
    def __init__(self, citys, demands, dim, capacity, trucks):
        self.citys = citys
        self.demands = demands
        self.dim = dim
        self.capacity = capacity
        self.trucks = trucks
        self.D = bm.zeros((citys.shape[0], citys.shape[0]))

    def calD(self):
        diff = self.citys[:, None, :] - self.citys[None, :, :]
        self.D = bm.sqrt(bm.sum(diff ** 2, axis=-1))
        self.D[bm.arange(self.dim), bm.arange(self.dim)] = 2.2204e-16

    def fitness(self, x):
        NP = bm.array(x).shape[0]
        sorted_numbers = bm.argsort(x)
        fit = bm.zeros(NP)  

        for pop_idx in range(NP):  
            allidx = sorted_numbers[pop_idx]

            total_demand, final_route = self.gbestroute_cvrp(allidx)

            if total_demand == sum(self.demands):
                fit[pop_idx] = self.cal_distance(final_route)
            else:
                fit[pop_idx] = self.cal_distance(final_route) + 100 # 添加惩罚  

        return fit
    
    def cal_distance(self, routes):
        distance = 0
        for route in routes:
            distance += sum(self.D[start, end] for start, end in zip(route[:-1], route[1:]))
        return distance 
    
    def gbestroute_cvrp(self, sorted_numbers):
        not_allowed = [0]
        allowed = [idx for idx in sorted_numbers if idx not in not_allowed]

        vehicle_route = [[] for _ in range(self.trucks)]  # 用于临时存储解析出的每辆车的路径
        sum_demands = []

        prev_vehicle_route = [[] for _ in range(self.trucks)] # 存上一次的路径
        prev_sum_demands = []

        routes = [] 
        demandlist = []
        truck_num = self.trucks
        while True:
            if len(allowed) > 0:
                for truck_idx in range(truck_num):
                    vehicle_route[truck_idx].append(allowed[truck_idx])
                sum_demands = bm.sum(self.demands[vehicle_route], axis=1)

                if (sum_demands > 160).any():
                    contiue_idx = bm.where(sum_demands <= 160)[0]
                    over_idx = bm.where(sum_demands > 160)[0]
                    if len(contiue_idx) > 0:
                        for oidx in over_idx:
                            routes.append(prev_vehicle_route[oidx])
                            demandlist.append(prev_sum_demands[oidx])
                            del vehicle_route[oidx]

                        for cidx in contiue_idx:
                            element = allowed[cidx]
                            not_allowed.append(element)

                        allowed = [idx for idx in sorted_numbers if idx not in not_allowed]
                        truck_num = truck_num - len(over_idx)
                    else:
                        routes.extend(vehicle_route)
                        demandlist.extend(sum_demands + 100) # 加100的惩罚
                        break
                else:   
                    prev_vehicle_route = [route.copy() for route in vehicle_route]
                    prev_sum_demands = sum_demands

                    not_allowed.extend(allowed[:truck_num])
                    allowed = [idx for idx in sorted_numbers if idx not in not_allowed]
            else:
                routes.extend(prev_vehicle_route) 
                demandlist.extend(prev_sum_demands) 
                break

        route = list(map(lambda x: [0] + x + [0], routes))
        total_demand = sum(demandlist)
        return total_demand, route       

def print_results_cvrp(results):
    for algo_name, result in results.items():
        print('The best optimal value by {0:5} \t'
              'is :{1:12.4f}\t'
              'Time :{2:8.4f} seconds'.format(algo_name, float(result['gbest_f']), result['time']))

    with open('results_CVRP.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Algorithm', 'Best Optimal Value', 'Time', 'Citys path', 'Citys path(x,y,z)']
        writer.writerow(header)
        for algo_name, result in results.items():
            writer.writerow([algo_name, result['gbest_f'], result['time'], result['route'], result['route_citys']])
    print("\nResults have been successfully saved to 'results_CVRP.csv'.")

def printroute_cvrp(citys, results, save_path="images_CVRP"):
    route_citys_all = [[[point for point in sub_route] for sub_route in result['route_citys']] for result in results.values()]
    alg_name = list(results.keys())

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    dim_citys = len(citys[0]) 
    if dim_citys == 2:
        for i in range(len(route_citys_all)):
            plt.figure()
            plt.title(f"{alg_name[i]} optimal path")
            plt.scatter([coord[0] for coord in citys], [coord[1] for coord in citys])
            for sub_route in route_citys_all[i]:
                x_coords = [point[0] for point in sub_route]
                y_coords = [point[1] for point in sub_route]
                plt.plot(x_coords, y_coords)

            figure_filename = f"{alg_name[i]}_optimal_path.png"
            plt.savefig(os.path.join(save_path, figure_filename))
            plt.close()

    elif dim_citys == 3:
        for i in range(len(route_citys_all)):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(f"{alg_name[i]} optimal path")
            ax.scatter([coord[0] for coord in citys], [coord[1] for coord in citys], [coord[2] for coord in citys])
            for sub_route in route_citys_all[i]:
                x_coords = [point[0] for point in sub_route]
                y_coords = [point[1] for point in sub_route]
                z_coords = [point[2] for point in sub_route]
                ax.plot(x_coords, y_coords, z_coords)            
            
            figure_filename = f"{alg_name[i]}_optimal_path.png"
            plt.savefig(os.path.join(save_path, figure_filename))
            plt.close() 
    print("Images have been successfully saved to 'images_CVRP'file.") 


if __name__ == "__main":
    pass
