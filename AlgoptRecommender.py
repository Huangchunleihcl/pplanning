import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
from datapre import MapDataPreprocessor

warnings.filterwarnings('ignore')

class AlgorithmRecommender:
    def __init__(self, data_path='optimal_algorithms.csv'):
        """
        初始化推荐系统
        :param data_path: 标签数据文件路径
        """
        self.data_path = data_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'obs_density', 
            'manhattan_dist',
            'nodes',
            'edges',
            'conn_comp',
            'edge_len'
        ]
        self.param_db = None
        self.label_encoder = None

    def load_data(self, test_size=0.3, random_state=42):
        """
        加载并预处理数据
        :param test_size: 测试集比例
        :return: 训练集和测试集元组 (X_train, X_test, y_train, y_test)
        """
        df = pd.read_csv(self.data_path)
        self.param_db = df.groupby('algorithm')['params'].agg(lambda x: x.mode()[0]).to_dict() # 构建参数数据库
        
        # 特征标准化
        X = df[self.feature_columns]
        y = df['algorithm']
        X = self.scaler.fit_transform(X)
        
        # 划分数据集
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def train(self, n_estimators=100, max_depth=None, verbose=True):
        """
        训练推荐模型
        :param n_estimators: 随机森林树数量
        :param max_depth: 最大树深度
        :param verbose: 是否显示训练信息
        :return: 训练好的模型
        """
        X_train, X_test, y_train, y_test = self.load_data() # 数据准备
        
        # 初始化模型
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=42
        )
        
        self.model.fit(X_train, y_train) # 训练模型
        
        # 评估模型
        if verbose:
            self.evaluate(X_test, y_test)
        
        return self.model

    def evaluate(self, X_test, y_test):
        """ 模型性能评估 """
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"模型评估结果：")
        print(f"- 准确率: {accuracy:.2%}")
        print(f"- 特征重要性：")
        for feat, imp in zip(self.feature_columns, self.model.feature_importances_):
            print(f"  {feat}: {imp:.2%}")

    def save_model(self, model_path='algo_recommender.pkl'):
        """ 保存完整推荐系统 """
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'param_db': self.param_db
        }, model_path)
        print(f"模型已保存至 {model_path}")

    @classmethod
    def load_model(cls, model_path='algo_recommender.pkl'):
        """ 加载预训练模型 """
        system = cls()
        data = joblib.load(model_path)
        system.model = data['model']
        system.scaler = data['scaler']
        system.param_db = data['param_db']
        return system

    def recommend(self, new_data, return_params=True):
        """
        为新地图数据推荐最佳算法
        :param new_data: 字典格式的特征数据
        :param return_params: 是否返回推荐参数
        :return: 推荐结果字典
        """
        features = pd.DataFrame([new_data])[self.feature_columns] # 转换为DataFrame
        scaled_features = self.scaler.transform(features) # 特征标准化
        
        algo = self.model.predict(scaled_features)[0] # 预测算法
        
        # 获取参数
        result = {'algorithm': algo}
        if return_params:
            result['params'] = json.loads(self.param_db[algo])
        
        return result



if __name__ == "__main__":
    # 训练并保存模型
    data_path = "Data/test/Tables/optimal_tables.csv"
    recommender = AlgorithmRecommender(data_path=data_path)
    recommender.train(n_estimators=150)
    recommender.save_model()

    # 加载预训练模型
    loaded_recommender = AlgorithmRecommender.load_model()

    
    # 新数据地图特征提取
    open_dir = 'Data/test'
    preprocessor = MapDataPreprocessor(open_dir)
    features = preprocessor.extract_features()
    new_map_data = list(features.values())[0]

    # 新数据预测
    recommendation = loaded_recommender.recommend(new_map_data)
    print("\n推荐结果：")
    print(f"最佳算法：{recommendation['algorithm']}")
    print("推荐参数：")
    print(json.dumps(recommendation['params'], indent=2))