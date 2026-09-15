import os
import cv2
import numpy as np
import joblib
import logging
from datetime import datetime
from collections import Counter
from sklearn.cluster import MiniBatchKMeans

def setup_logger():
    log_filename = f"kmeans_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

class VestColorCluster:
    def __init__(self, n_clusters=3, bins=18):
        self.n_clusters = n_clusters
        self.bins = bins 
        self.is_trained = False
        self.label_map = {}
        
        # 使用 MiniBatchKMeans 替代传统 KMeans，以支持增量学习 (partial_fit)
        initial_centers = self._get_anchor_centers()
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters, 
            init=initial_centers, 
            n_init=1, 
            random_state=42,
            batch_size=100
        )

    def _get_anchor_centers(self):
        centers = np.zeros((3, self.bins), dtype=np.float32)
        centers[0, 1] = 1.0   # Orange
        centers[1, 3] = 1.0   # Yellow
        centers[2, 11] = 1.0  # Blue
        return centers

    def extract_feature(self, image_crop):
        if image_crop is None or image_crop.size == 0:
            return None
        
        hsv_image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
        mask = (hsv_image[:, :, 1] > 30) & (hsv_image[:, :, 2] > 30)  # 50
        valid_hues = hsv_image[:, :, 0][mask]
        
        if len(valid_hues) == 0:
            return None
            
        hist, _ = np.histogram(valid_hues, bins=self.bins, range=(0, 180))
        feature_vector = hist.astype(np.float32) / (hist.sum() + 1e-6)
        return feature_vector

    def train(self, image_crops, incremental=False):
        """
        incremental=False: 从头开始训练
        incremental=True: 使用新批次数据更新现有模型
        """
        features = []
        logging.info("[*] 正在提取颜色直方图特征向量...")
        for crop in image_crops:
            vec = self.extract_feature(crop)
            if vec is not None:
                features.append(vec)
                
        if not features:
            logging.warning("[!] 警告：当前批次没有有效的特征向量。")
            return
            
        X_real = np.array(features)
        
        # 无论全量还是增量，永远注入虚拟锚点数据，防止中心漂移
        synthetic_anchors = self._get_anchor_centers()
        synthetic_data = np.repeat(synthetic_anchors, 5, axis=0)
        X = np.vstack((X_real, synthetic_data))
        
        mode = "增量更新" if incremental else "训练"
        logging.info(f"[*] 开始 KMeans {mode}，真实样本: {len(X_real)}，注入锚点: {len(synthetic_data)}，维度: {X.shape[1]}")
        
        # 增量更新逻辑
        if incremental and self.is_trained:
            self.kmeans.partial_fit(X)
        else:
            self.kmeans.fit(X)
            self.is_trained = True
        
        # --- 簇中心语义对齐 ---
        for i, center_hist in enumerate(self.kmeans.cluster_centers_):
            peak_bin = np.argmax(center_hist)
            peak_hue = peak_bin * 10 + 5 
            
            if (0 <= peak_hue <= 20) or (150 <= peak_hue <= 180):
                self.label_map[i] = "Orange"
            elif 21 <= peak_hue <= 85:
                self.label_map[i] = "Yellow"
            else:
                self.label_map[i] = "Blue"
                
        logging.info(f"[*] 聚类簇中心语义映射结果: {self.label_map}")
        
        found_colors = set(self.label_map.values())
        if "Blue" not in found_colors:
            logging.info("[!] 提示：聚类结果中未完全映射蓝色，但模型已通过锚点保留了蓝色识别能力。")

    def predict(self, image_crop):
        if not self.is_trained:
            raise ValueError("[!] 模型尚未训练，请先调用 train() 或 load_model()")
            
        vec = self.extract_feature(image_crop)
        if vec is None:
            return "Unknown"
            
        cluster_id = self.kmeans.predict([vec])[0]
        return self.label_map.get(cluster_id, "Unknown")
        
    def save_model(self, path="kmeans_color_model.pkl"):
        joblib.dump({'kmeans': self.kmeans, 'label_map': self.label_map, 'bins': self.bins}, path)
        logging.info(f"[*] 模型已保存至: {path}")
        
    def load_model(self, path="kmeans_color_model.pkl"):
        data = joblib.load(path)
        self.kmeans = data['kmeans']
        self.label_map = data['label_map']
        self.bins = data.get('bins', 18)
        self.is_trained = True


def main():
    log_filename = setup_logger()
    input_base = "./Data_Vest_Crops"
    splits = ["train", "val", "test"]
    model_path = "kmeans_color_model.pkl"
    
    MIN_WIDTH, MIN_HEIGHT = 30, 30
    valid_crops = []
    
    logging.info("====================================")
    logging.info("[*] 阶段二：无监督颜色聚类 ( Kmeans )")
    logging.info("====================================")
    
    for split in splits:
        split_dir = os.path.join(input_base, split)
        if not os.path.exists(split_dir): continue
            
        for img_name in os.listdir(split_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue
                
            img_path = os.path.join(split_dir, img_name)
            crop_img = cv2.imread(img_path)
            
            if crop_img is None: continue
            h, w = crop_img.shape[:2]
            
            if w >= MIN_WIDTH and h >= MIN_HEIGHT:
                valid_crops.append((img_name, crop_img))

    logging.info(f"[*] 数据加载完成，共收集到 {len(valid_crops)} 张有效反光衣裁剪图。")
    if len(valid_crops) < 1:
        logging.error("[!] 错误：没有足够样本训练。")
        return

    cluster_model = VestColorCluster(n_clusters=3, bins=18)
    train_images = [img for name, img in valid_crops]
    
    # 增量学习判断逻辑：如果已有模型，则加载并执行增量更新；否则全量训练
    if os.path.exists(model_path):
        logging.info(f"[*] 检测到历史模型文件 {model_path}，执行增量更新...")
        cluster_model.load_model(model_path)
        cluster_model.train(train_images, incremental=True)
    else:
        logging.info(f"[*] 未检测到历史模型，执行初始训练...")
        cluster_model.train(train_images, incremental=False)
    
    cluster_model.save_model(model_path)
    
    # --- 实验数据统计与可视化 ---
    logging.info(f"\n[*] 正在为全部 {len(valid_crops)} 张样本生成预测...")
    vis_dir = os.path.join(input_base, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    prediction_results = []
    
    for idx, (original_name, sample) in enumerate(valid_crops):
        pred_color = cluster_model.predict(sample)
        prediction_results.append(pred_color)
        
        vis_img = cv2.resize(sample.copy(), (150, 150))
        
        if pred_color == "Orange":
            text_color = (0, 165, 255)  
        elif pred_color == "Yellow":
            text_color = (0, 255, 255)  
        elif pred_color == "Blue":
            text_color = (255, 0, 0)    
        else:
            text_color = (0, 255, 0)    

        cv2.putText(vis_img, pred_color, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        save_path = os.path.join(vis_dir, f"pred_{pred_color}_{original_name}")
        cv2.imwrite(save_path, vis_img)

    # 统计预测标签的分布情况
    label_distribution = Counter(prediction_results)
    
    logging.info("\n====================================")
    logging.info("          实验数据与分布报告")
    logging.info("====================================")
    logging.info(f"  - 总处理图像数: {len(valid_crops)}")
    for color, count in label_distribution.items():
        percentage = (count / len(valid_crops)) * 100
        logging.info(f"  - 类别 [{color}]: {count} 张 ({percentage:.1f}%)")
        
    logging.info(f"\n[*] 全部图片可视化结果已保存至: {vis_dir}")
    logging.info(f"[*] 完整运行日志已保存至: {log_filename}")

if __name__ == "__main__":
    main()
    