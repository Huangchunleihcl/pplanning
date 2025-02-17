import joblib
import pandas as pd
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from main import ShortestPathApp
from AGV_maps import short_folder
from BayesOpt import extract_map_features

class AlgorithmRecommender:
    def __init__(self, historical_data_path="historical_results"):
        """
        :param historical_data_path: 历史结果存储目录
        """
        self.historical_data = self._load_historical_data(historical_data_path)
        self.model = None
        self.algorithm_best_params = {}
        self.feature_columns = [
            'obs_density', 'manhattan_dist', 'nodes',
            'edges', 'conn_comp', 'edge_len'
        ]
        
    def _load_historical_data(self, path):
        """加载并预处理历史数据"""
        all_files = list(Path(path).glob("*.csv"))
        df = pd.concat([pd.read_csv(f) for f in all_files])
        
        # 为每个地图保留最优记录
        best_indices = df.groupby(['map_id'])['composite'].idxmin()
        return df.loc[best_indices].reset_index(drop=True)

    def train_model(self):
        """训练单级推荐模型"""
        # 准备训练数据
        X = self.historical_data[self.feature_columns]
        y = self.historical_data['algo']
        
        # 训练分类模型
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        
        # 提取各算法全局最优参数
        for algo in self.historical_data['algo'].unique():
            algo_data = self.historical_data[self.historical_data['algo'] == algo]
            best_idx = algo_data['composite'].idxmin()
            self.algorithm_best_params[algo] = json.loads(
                algo_data.loc[best_idx, 'params']
            )
        
        # 保存模型
        joblib.dump((self.model, self.algorithm_best_params), 
                   'recommender_model.pkl')

    def recommend(self, map_data, start, goal):
        """推荐最优算法及参数"""
        features = extract_map_features(map_data, start, goal)
        features_df = pd.DataFrame([features[self.feature_columns]])
        
        # 预测最优算法
        algo = self.model.predict(features_df)[0]
        
        return {
            'algorithm': algo,
            'parameters': self.algorithm_best_params.get(algo, {}),
            'features': features
        }

    def execute_recommendation(self, recommendation, map_data):
        """执行推荐配置"""
        app = ShortestPathApp(
            map_data,
            recommendation['features']['start'],
            recommendation['features']['goal'],
            algo_chosen=[recommendation['algorithm']],
            **recommendation['parameters']
        )
        
        results = app.optimize()
        return {
            'path': results['route'],
            'length': results['gbest_f'],
            'turns': results['truns'],
            'time': results['time']
        }

# 使用示例
if __name__ == "__main__":
    # 初始化推荐系统
    recommender = AlgorithmRecommender("historical_results")
    
    # 训练模型（首次需要运行）
    if not Path("recommender_model.pkl").exists():
        recommender.train_model()
    else:
        model, params = joblib.load("recommender_model.pkl")
        recommender.model = model
        recommender.algorithm_best_params = params
    
    # 新地图数据
    new_map = short_folder("new_map.map")
    
    # 生成推荐
    recommendation = recommender.recommend(
        new_map['maps'],
        new_map['start'],
        new_map['goal']
    )
    
    # 执行推荐配置
    results = recommender.execute_recommendation(recommendation, new_map)
    
    print("推荐配置：")
    print(f"算法：{recommendation['algorithm']}")
    print(f"参数：{json.dumps(recommendation['parameters'], indent=2)}")
    
    print("\n执行结果：")
    print(f"路径长度：{results['length']:.2f}")
    print(f"转弯次数：{results['turns']}")
    print(f"计算时间：{results['time']:.2f}s")