import os
import cv2
import csv
import numpy as np
import joblib
import logging
from datetime import datetime
from collections import Counter
from sklearn.svm import SVC

def setup_logger():
    log_filename = f"svm_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        force=True,
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_filename

class VestColorClassifier:
    def __init__(self, bins=18):
        self.bins = bins 
        self.is_trained = False
        # 使用 RBF 核函数的 SVM，自动处理类别不平衡
        self.model = SVC(kernel='rbf', class_weight='balanced', random_state=42)

    def extract_feature(self, image_crop):
        if image_crop is None or image_crop.size == 0:
            return None
        
        hsv_image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
        # 【特征提纯】：将亮度和饱和度门槛设为30，滤除阴影和暗淡像素
        mask = (hsv_image[:, :, 1] > 30) & (hsv_image[:, :, 2] > 30)
        valid_hues = hsv_image[:, :, 0][mask]
        
        if len(valid_hues) == 0:
            return None
            
        hist, _ = np.histogram(valid_hues, bins=self.bins, range=(0, 180))
        return hist.astype(np.float32) / (hist.sum() + 1e-6)

    def train(self, image_crops, labels):
        """
        有监督训练核心：接收图片及其对应的人工标签 (y)
        """
        features = []
        valid_labels = []
        
        logging.info("[*] 正在提取颜色直方图特征向量...")
        for crop, label in zip(image_crops, labels):
            vec = self.extract_feature(crop)
            if vec is not None:
                features.append(vec)
                valid_labels.append(label)
                
        if not features:
            logging.error("[!] 错误：没有提取到有效特征。")
            return
            
        X = np.array(features)
        y = np.array(valid_labels)
        
        logging.info(f"[*] 开始 SVM 有监督训练，样本数量: {len(X)}，特征维度: {X.shape[1]}")
        self.model.fit(X, y)
        self.is_trained = True
        
        # 统计模型实际学习到的类别
        learned_classes = self.model.classes_
        logging.info(f"[*] 模型已成功拟合，当前掌握的分类器边界包含: {learned_classes}")

    def predict(self, image_crop):
        if not self.is_trained:
            raise ValueError("[!] 模型尚未训练！")
            
        vec = self.extract_feature(image_crop)
        if vec is None:
            # 对于全黑无效图片，默认给个最常见的类兜底
            return "Orange" 
            
        return self.model.predict([vec])[0]
        
    def save_model(self, path="svm_color_model.pkl"):
        joblib.dump({'model': self.model, 'bins': self.bins}, path)
        logging.info(f"[*] 模型已保存至: {path}")
        
    def load_model(self, path="svm_color_model.pkl"):
        data = joblib.load(path)
        self.model = data['model']
        self.bins = data.get('bins', 18)
        self.is_trained = True

def main():
    """
    当单独运行时，它的职责是：读取完整的 ground_truth_labels.csv，
    利用 100% 的数据训练出一个模型，并保存。
    """
    log_filename = setup_logger()
    csv_path = "./ground_truth_labels.csv"
    model_path = "svm_color_model.pkl"
    
    logging.info("====================================")
    logging.info("[*] 阶段二：有监督颜色分类模块 (SVM )")
    logging.info("====================================")
    
    if not os.path.exists(csv_path):
        logging.error(f"[!] 找不到 {csv_path}，请先运行 3.py 生成模板并打标。")
        return

    X_images = []
    y_true = []
    
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            true_color = row["True_Color"].strip()
            if true_color not in ["Orange", "Yellow", "Blue"]: 
                continue
            img = cv2.imread(row["File_Path"])
            if img is not None:
                X_images.append(img)
                y_true.append(true_color)

    if not X_images:
        logging.error("[!] 未读取到有效标注数据。")
        return

    classifier = VestColorClassifier(bins=18)
    classifier.train(X_images, y_true)
    classifier.save_model(model_path)
    
    label_distribution = Counter(y_true)
    logging.info(f"\n[*] 模型训练完成。数据分布: {dict(label_distribution)}")

if __name__ == "__main__":
    main()
    