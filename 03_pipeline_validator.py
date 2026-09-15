import os
import cv2
import csv
import numpy as np
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
from sklearn.svm import SVC
from sklearn.cluster import KMeans

def setup_logger():
    log_filename = f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

def extract_feature(image_crop, bins=18):
    """独立的特征提取模块"""
    if image_crop is None or image_crop.size == 0:
        return None
    hsv_image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
    mask = (hsv_image[:, :, 1] > 30) & (hsv_image[:, :, 2] > 30)
    valid_hues = hsv_image[:, :, 0][mask]
    if len(valid_hues) == 0:
        return None
    hist, _ = np.histogram(valid_hues, bins=bins, range=(0, 180))
    return hist.astype(np.float32) / (hist.sum() + 1e-6)

def plot_confusion_matrices(y_true, y_pred_kmeans, y_pred_svm, classes, save_dir):
    """绘制并列的混淆矩阵"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    cm_kmeans = confusion_matrix(y_true, y_pred_kmeans, labels=classes)
    sns.heatmap(cm_kmeans, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=classes, yticklabels=classes)
    axes[0].set_title('KMeans Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    cm_svm = confusion_matrix(y_true, y_pred_svm, labels=classes)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', ax=axes[1], 
                xticklabels=classes, yticklabels=classes)
    axes[1].set_title('SVM Confusion Matrix')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

def plot_metrics_comparison(metrics_dict, save_dir):
    """绘制量化指标对比柱状图"""
    labels = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)']
    kmeans_scores = [
        metrics_dict['KMeans']['accuracy'],
        metrics_dict['KMeans']['precision'],
        metrics_dict['KMeans']['recall'],
        metrics_dict['KMeans']['f1']
    ]
    svm_scores = [
        metrics_dict['SVM']['accuracy'],
        metrics_dict['SVM']['precision'],
        metrics_dict['SVM']['recall'],
        metrics_dict['SVM']['f1']
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, kmeans_scores, width, label='KMeans (Unsupervised)', color='#5DADE2')
    rects2 = ax.bar(x + width/2, svm_scores, width, label='SVM (Supervised)', color='#48C9B0')

    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison (Test Set)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim([0, 1.1])
    ax.legend(loc='lower right')

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path

def train_and_predict_kmeans(X_train, X_test):
    """KMeans 训练与预测逻辑"""
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_train)
    
    label_map = {}
    for i, center_hist in enumerate(kmeans.cluster_centers_):
        peak_bin = np.argmax(center_hist)
        peak_hue = peak_bin * 10 + 5 
        if 0 <= peak_hue <= 25:
            label_map[i] = "Orange"
        elif 26 <= peak_hue <= 85:
            label_map[i] = "Yellow"
        else:
            label_map[i] = "Blue"
            
    # 如果类别分配有缺失，使用 fallback 防止报错
    y_pred = []
    for x in X_test:
        cluster_id = kmeans.predict([x])[0]
        y_pred.append(label_map.get(cluster_id, "Orange"))
    return y_pred

def train_and_predict_svm(X_train, y_train, X_test):
    """SVM 训练与预测逻辑"""
    svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    svm.fit(X_train, y_train)
    return svm.predict(X_test)

def main():
    log_filename = setup_logger()
    csv_path = "./ground_truth_labels.csv"
    output_vis_dir = "./evaluation_results"
    os.makedirs(output_vis_dir, exist_ok=True)
    
    logging.info("==================================================")
    logging.info("   阶段三：KMeans 与 SVM 核心算法平行对比验证")
    logging.info("==================================================")
    
    if not os.path.exists(csv_path):
        logging.error(f"[!] 找不到 {csv_path}，请确保已提供标注数据。")
        return

    X_features = []
    y_true = []
    
    with open(csv_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            true_color = row["True_Color"].strip()
            if true_color not in ["Orange", "Yellow", "Blue"]:
                continue
                
            img = cv2.imread(row["File_Path"])
            vec = extract_feature(img)
            if vec is not None:
                X_features.append(vec)
                y_true.append(true_color)

    if not X_features:
        logging.error("[!] 未提取到有效特征。")
        return

    X_features = np.array(X_features)
    y_true = np.array(y_true)
    classes = ["Orange", "Yellow", "Blue"]

    # 3:7 数据拆分
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_true, test_size=0.7, random_state=42, stratify=y_true
    )
    
    logging.info(f"[*] 数据集加载完毕。总有效样本: {len(X_features)} (训练集: {len(X_train)}, 测试集: {len(X_test)})")

    # 1. 运行 KMeans
    logging.info("[*] 正在执行 KMeans (无监督) 评估...")
    y_pred_kmeans = train_and_predict_kmeans(X_train, X_test)
    
    # 2. 运行 SVM
    logging.info("[*] 正在执行 SVM (有监督) 评估...")
    y_pred_svm = train_and_predict_svm(X_train, y_train, X_test)

    # 3. 计算量化指标
    def get_metrics(y_t, y_p):
        acc = accuracy_score(y_t, y_p)
        p, r, f1, _ = precision_recall_fscore_support(y_t, y_p, average='macro', zero_division=0)
        return {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}

    metrics_dict = {
        'KMeans': get_metrics(y_test, y_pred_kmeans),
        'SVM': get_metrics(y_test, y_pred_svm)
    }

    # 4. 生成可视化报告
    logging.info("[*] 正在生成混淆矩阵与性能对比图表...")
    cm_path = plot_confusion_matrices(y_test, y_pred_kmeans, y_pred_svm, classes, output_vis_dir)
    bar_path = plot_metrics_comparison(metrics_dict, output_vis_dir)

    logging.info("\n==================================================")
    logging.info("                KMeans 性能报告")
    logging.info("==================================================")
    for line in classification_report(y_test, y_pred_kmeans, zero_division=0).split('\n'):
        logging.info(line)

    logging.info("\n==================================================")
    logging.info("                 SVM 性能报告")
    logging.info("==================================================")
    for line in classification_report(y_test, y_pred_svm, zero_division=0).split('\n'):
        logging.info(line)

    logging.info("\n==================================================")
    logging.info(f" [图表输出] 混淆矩阵保存至: {cm_path}")
    logging.info(f" [图表输出] 指标对比柱状图保存至: {bar_path}")
    logging.info("==================================================")

if __name__ == "__main__":
    main()
    