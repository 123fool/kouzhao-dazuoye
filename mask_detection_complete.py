# -*- coding: utf-8 -*-
"""
口罩检测课程作业 - 完整可运行版本
包含：
1. 数据集构建（从XML标注构建分类数据集）
2. SVM模型训练（经典机器学习）
3. ResNet18模型训练（深度学习）
4. 完整的性能评估和可视化

使用说明：
1. 修改 ROOT 变量为你的数据路径
2. 运行：python mask_detection_complete.py
"""

import os
import sys
import shutil
import random
import xml.etree.ElementTree as ET
import time
import copy
from collections import defaultdict

# 延迟导入numpy以避免DLL加载问题
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None
    print("警告: numpy 不可用")

from PIL import Image

# 深度学习库
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch 不可用，将跳过深度学习模型")

# 机器学习库
try:
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from joblib import dump, load
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: scikit-learn 不可用，将跳过SVM模型")

# 图像处理库
try:
    from skimage.io import imread
    from skimage.color import rgb2gray
    from skimage.transform import resize
    from skimage.feature import hog
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("警告: scikit-image 不可用")

# 可视化库（可选）
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib 不可用，将跳过可视化")

# ========================= 配置部分 =========================

# 🔴 重要：修改这里为你的数据路径
ROOT = r"E:\WorkSpace-Jiang\口罩\面部口罩检测数据集"  # 改为你的数据路径

IMG_DIR = os.path.join(ROOT, "images")
ANN_DIR = os.path.join(ROOT, "annotations")
DATASET_DIR = "dataset"
MODELS_DIR = "models"

# 数据集配置
random.seed(42)
if NUMPY_AVAILABLE and np is not None:
    np.random.seed(42)
if TORCH_AVAILABLE:
    torch.manual_seed(42)

CLASS_NAMES = ["with_mask", "without_mask", "mask_incorrect"]

# SVM和HOG参数
IMG_SIZE_HOG = (128, 128)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)

# ========================= 第1部分：数据集构建 =========================

def parse_annotation(xml_file):
    """解析VOC格式XML文件"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        objects = []
        for obj in root.iter("object"):
            name = obj.find("name").text
            
            # 标准化类名
            if name == "with_mask":
                cls = "with_mask"
            elif name == "without_mask":
                cls = "without_mask"
            else:
                cls = "mask_incorrect"
            
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            
            objects.append((cls, xmin, ymin, xmax, ymax))
        
        filename_node = root.find("filename")
        filename = filename_node.text if filename_node is not None else None
        
        return filename, objects
    except Exception as e:
        print(f"解析失败 {xml_file}: {e}")
        return None, []


def build_classification_dataset():
    """构建分类数据集：从XML+原图裁剪人脸，按7:1.5:1.5划分"""
    print("\n" + "="*60)
    print("第1步：构建分类数据集")
    print("="*60)
    
    if not os.path.exists(IMG_DIR) or not os.path.exists(ANN_DIR):
        print(f"❌ 错误: 数据路径不存在！")
        print(f"   IMG_DIR: {IMG_DIR}")
        print(f"   ANN_DIR: {ANN_DIR}")
        print(f"请修改 ROOT 变量为正确的数据路径")
        return False
    
    # 删除旧数据集
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    
    # 创建目录结构
    for split in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(DATASET_DIR, split, cls), exist_ok=True)
    
    # 解析所有XML并裁剪人脸
    all_crops = []
    xml_files = [f for f in os.listdir(ANN_DIR) if f.endswith('.xml')]
    
    print(f"发现 {len(xml_files)} 个XML标注文件")
    
    for i, xml_name in enumerate(xml_files):
        if (i + 1) % 100 == 0:
            print(f"  处理中... {i+1}/{len(xml_files)}")
        
        xml_path = os.path.join(ANN_DIR, xml_name)
        filename, objects = parse_annotation(xml_path)
        
        if not filename:
            continue
        
        # 查找对应的图像文件
        img_path = os.path.join(IMG_DIR, filename)
        if not os.path.exists(img_path):
            # 尝试其他格式
            base = os.path.splitext(filename)[0]
            found = False
            for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
                alt = os.path.join(IMG_DIR, base + ext)
                if os.path.exists(alt):
                    img_path = alt
                    found = True
                    break
            if not found:
                continue
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  ⚠ 无法打开图像: {img_path}")
            continue
        
        img_w, img_h = img.size
        ext = os.path.splitext(img_path)[1]
        
        # 裁剪每个检测到的人脸
        for idx, (cls, xmin, ymin, xmax, ymax) in enumerate(objects):
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_w - 1, xmax)
            ymax = min(img_h - 1, ymax)
            
            if xmax <= xmin or ymax <= ymin:
                continue
            
            crop = img.crop((xmin, ymin, xmax, ymax))
            save_name = f"{os.path.splitext(xml_name)[0]}_{idx}{ext}"
            all_crops.append((crop, cls, save_name))
    
    if len(all_crops) == 0:
        print("❌ 错误: 没有解析到任何数据，请检查数据路径！")
        return False
    
    # 数据划分
    random.shuffle(all_crops)
    n = len(all_crops)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    
    train_set = all_crops[:n_train]
    val_set = all_crops[n_train:n_train + n_val]
    test_set = all_crops[n_train + n_val:]
    
    # 保存数据集
    split_data = {"train": train_set, "val": val_set, "test": test_set}
    for split, samples in split_data.items():
        for crop, cls, name in samples:
            save_path = os.path.join(DATASET_DIR, split, cls, name)
            crop.save(save_path)
    
    # 统计信息
    print("\n✓ 数据集构建完成！")
    for split in ["train", "val", "test"]:
        total = sum(
            len(os.listdir(os.path.join(DATASET_DIR, split, cls)))
            for cls in CLASS_NAMES
        )
        print(f"  {split.upper():5s} 集: {total:4d} 样本", end="")
        
        for cls in CLASS_NAMES:
            count = len(os.listdir(os.path.join(DATASET_DIR, split, cls)))
            print(f"  | {cls}: {count:3d}", end="")
        print()
    
    return True


# ========================= 第2部分：SVM模型 =========================

def load_hog_features(split):
    """提取HOG特征"""
    X, y = [], []
    
    for label_idx, cls in enumerate(CLASS_NAMES):
        folder = os.path.join(DATASET_DIR, split, cls)
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
        
        for fname in files:
            try:
                path = os.path.join(folder, fname)
                img = imread(path)
                
                if img.ndim == 3:
                    img_gray = rgb2gray(img)
                else:
                    img_gray = img
                
                img_resized = resize(img_gray, IMG_SIZE_HOG, anti_aliasing=True)
                feat = hog(img_resized, **HOG_PARAMS)
                X.append(feat)
                y.append(label_idx)
            except Exception as e:
                pass
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def train_svm():
    """训练SVM模型"""
    if not SKLEARN_AVAILABLE or not SKIMAGE_AVAILABLE:
        print("\n⚠ SVM模型跳过: 缺少必要的库")
        return None
    
    print("\n" + "="*60)
    print("第2步：训练SVM模型 (HOG + SVM)")
    print("="*60)
    
    print("提取HOG特征中...")
    X_train, y_train = load_hog_features("train")
    X_val, y_val = load_hog_features("val")
    X_test, y_test = load_hog_features("test")
    
    print(f"✓ 特征提取完成")
    print(f"  HOG特征维度: {X_train.shape[1]}")
    print(f"  训练样本: {X_train.shape[0]}")
    
    print("\n训练SVM模型中...")
    svm = SVC(kernel="rbf", C=10, gamma="scale", verbose=1)
    svm.fit(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]))
    
    # 评估
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ SVM训练完成！")
    print(f"  测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 按类别统计
    print("\n  按类别准确率:")
    for i, cls_name in enumerate(CLASS_NAMES):
        mask = y_test == i
        if np.sum(mask) > 0:
            acc = np.sum(y_pred[mask] == y_test[mask]) / np.sum(mask)
            print(f"    {cls_name:20s}: {acc:.4f}")
    
    # 保存模型
    os.makedirs(MODELS_DIR, exist_ok=True)
    dump(svm, os.path.join(MODELS_DIR, "svm_hog_mask.joblib"))
    print(f"\n✓ 模型已保存: {MODELS_DIR}/svm_hog_mask.joblib")
    
    return {"name": "SVM", "accuracy": accuracy, "y_true": y_test, "y_pred": y_pred}


# ========================= 第3部分：深度学习模型 =========================

def train_resnet18(num_epochs=5):
    """训练ResNet18模型"""
    if not TORCH_AVAILABLE:
        print("\n⚠ ResNet18模型跳过: PyTorch不可用")
        return None
    
    print("\n" + "="*60)
    print("第3步：训练ResNet18模型 (深度学习)")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据变换
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # 加载数据集
    image_datasets = {
        x: datasets.ImageFolder(
            root=os.path.join(DATASET_DIR, x),
            transform=data_transforms[x]
        )
        for x in ["train", "val", "test"]
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=32, shuffle=(x == "train"), num_workers=0)
        for x in ["train", "val", "test"]
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}
    print(f"\n数据集大小: {dataset_sizes}")
    
    # 构建模型
    print("\n加载预训练ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 冻结特征层，只训练分类层
    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    model = model.to(device)
    
    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 训练循环
    print(f"\n开始训练 ({num_epochs} epochs)...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == "train":
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f"  {phase.upper():5s} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
            
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    # 加载最佳权重
    model.load_state_dict(best_model_wts)
    
    # 测试评估
    print("\n" + "-"*60)
    print("在测试集上评估...")
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    accuracy = np.sum(all_preds == all_labels) / len(all_labels)
    
    print(f"\n✓ ResNet18训练完成！")
    print(f"  测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 按类别统计
    print("\n  按类别准确率:")
    for i, cls_name in enumerate(CLASS_NAMES):
        mask = all_labels == i
        if np.sum(mask) > 0:
            acc = np.sum(all_preds[mask] == all_labels[mask]) / np.sum(mask)
            print(f"    {cls_name:20s}: {acc:.4f}")
    
    # 保存模型
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "resnet18_mask.pth"))
    print(f"\n✓ 模型已保存: {MODELS_DIR}/resnet18_mask.pth")
    
    return {"name": "ResNet18", "accuracy": accuracy, "y_true": all_labels, "y_pred": all_preds}


# ========================= 第4部分：性能对比与可视化 =========================

def print_results_summary(results):
    """打印结果总结"""
    print("\n" + "="*60)
    print("最终结果总结")
    print("="*60)
    
    if results:
        print("\n模型准确率对比:")
        for result in results:
            if result:
                print(f"  {result['name']:15s}: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
        
        if len(results) == 2 and results[0] and results[1]:
            diff = abs(results[1]['accuracy'] - results[0]['accuracy'])
            better = results[1]['name'] if results[1]['accuracy'] > results[0]['accuracy'] else results[0]['name']
            print(f"\n  性能差异: {diff:.4f}")
            print(f"  更优秀的模型: {better}")


def plot_confusion_matrix(y_true, y_pred, model_name):
    """简单的混淆矩阵打印"""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n{model_name} 混淆矩阵:")
    print("           预测", "  ".join([f"{cls:10s}" for cls in CLASS_NAMES]))
    for i, cls_name in enumerate(CLASS_NAMES):
        print(f"真实 {cls_name:10s}: {' '.join([f'{cm[i,j]:5d}' for j in range(len(CLASS_NAMES))])}")


def visualize_results(results):
    """生成结果可视化图表"""
    if not MATPLOTLIB_AVAILABLE:
        print("\n⚠ matplotlib不可用，跳过可视化")
        return
    
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    
    print("\n" + "="*60)
    print("生成可视化图表...")
    print("="*60)
    
    # 设置图表风格
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # ============ 图表1: 模型准确率对比 ============
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    model_names = []
    accuracies = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, result in enumerate(results):
        if result:
            model_names.append(result['name'])
            accuracies.append(result['accuracy'])
    
    bars = ax1.bar(model_names, accuracies, color=colors[:len(model_names)], 
                   alpha=0.7, edgecolor='black', linewidth=2)
    
    # 添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.4f}\n({acc*100:.2f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('准确率 (Accuracy)', fontsize=12, fontweight='bold')
    ax1.set_title('模型准确率对比', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('图表1_模型准确率对比.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: 图表1_模型准确率对比.png")
    plt.close()
    
    # ============ 图表2: 各类别准确率对比 ============
    if len(results) > 0 and results[0]:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        
        model_names_detail = []
        class_accuracies = {cls: [] for cls in CLASS_NAMES}
        
        for result in results:
            if result:
                model_names_detail.append(result['name'])
                y_true = result['y_true']
                y_pred = result['y_pred']
                
                for i, cls_name in enumerate(CLASS_NAMES):
                    mask = y_true == i
                    if np.sum(mask) > 0:
                        acc = np.sum(y_pred[mask] == y_true[mask]) / np.sum(mask)
                        class_accuracies[cls_name].append(acc)
                    else:
                        class_accuracies[cls_name].append(0)
        
        x = np.arange(len(model_names_detail))
        width = 0.25
        
        for i, cls_name in enumerate(CLASS_NAMES):
            offset = (i - 1) * width
            bars = ax2.bar(x + offset, class_accuracies[cls_name], width,
                          label=cls_name, alpha=0.8, edgecolor='black')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom', fontsize=9)
        
        ax2.set_xlabel('模型', fontsize=12, fontweight='bold')
        ax2.set_ylabel('准确率 (Accuracy)', fontsize=12, fontweight='bold')
        ax2.set_title('各类别准确率对比', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names_detail)
        ax2.legend(loc='lower right', fontsize=11)
        ax2.set_ylim(0, 1.1)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('图表2_各类别准确率对比.png', dpi=300, bbox_inches='tight')
        print("✓ 已保存: 图表2_各类别准确率对比.png")
        plt.close()
    
    # ============ 图表3: 混淆矩阵热力图 ============
    fig3, axes = plt.subplots(1, len([r for r in results if r]), 
                             figsize=(6*len([r for r in results if r]), 5))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    for idx, result in enumerate([r for r in results if r]):
        y_true = result['y_true']
        y_pred = result['y_pred']
        model_name = result['name']
        
        cm = confusion_matrix(y_true, y_pred)
        
        # 归一化混淆矩阵（按行）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                   ax=axes[idx], cbar=True, annot_kws={'size': 10})
        
        axes[idx].set_title(f'{model_name} 混淆矩阵\n(百分比)', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('预测标签', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel('真实标签', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('图表3_混淆矩阵热力图.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: 图表3_混淆矩阵热力图.png")
    plt.close()
    
    # ============ 图表4: 数据集分布（训练集） ============
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4))
    
    splits = ['train', 'val', 'test']
    colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for ax, split in zip(axes4, splits):
        counts = []
        for cls in CLASS_NAMES:
            folder = os.path.join(DATASET_DIR, split, cls)
            if os.path.exists(folder):
                count = len([f for f in os.listdir(folder) 
                           if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
                counts.append(count)
            else:
                counts.append(0)
        
        wedges, texts, autotexts = ax.pie(counts, labels=CLASS_NAMES, autopct='%1.1f%%',
                                          colors=colors_pie, startangle=90,
                                          textprops={'fontsize': 10, 'fontweight': 'bold'})
        
        # 添加数值
        total = sum(counts)
        ax.set_title(f'{split.upper()} 集分布\n(总数: {total})', 
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('图表4_数据集分布.png', dpi=300, bbox_inches='tight')
    print("✓ 已保存: 图表4_数据集分布.png")
    plt.close()
    
    # ============ 图表5: 详细的分类报告（仅显示第一个模型） ============
    if results[0]:
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        
        y_true = results[0]['y_true']
        y_pred = results[0]['y_pred']
        model_name = results[0]['name']
        
        # 获取分类报告的各项指标
        report = classification_report(y_true, y_pred, output_dict=True, 
                                      target_names=CLASS_NAMES)
        
        metrics = ['precision', 'recall', 'f1-score']
        x = np.arange(len(CLASS_NAMES))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [report[cls][metric] for cls in CLASS_NAMES]
            offset = (i - 1) * width
            bars = ax5.bar(x + offset, values, width, label=metric.upper(),
                          alpha=0.8, edgecolor='black')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9)
        
        ax5.set_xlabel('类别', fontsize=12, fontweight='bold')
        ax5.set_ylabel('分数', fontsize=12, fontweight='bold')
        ax5.set_title(f'{model_name} 模型详细评估指标', fontsize=14, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(CLASS_NAMES)
        ax5.legend(loc='lower right', fontsize=11)
        ax5.set_ylim(0, 1.1)
        ax5.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('图表5_分类评估指标.png', dpi=300, bbox_inches='tight')
        print("✓ 已保存: 图表5_分类评估指标.png")
        plt.close()
    
    print("\n✓ 所有图表生成完成！")
    print("  生成的图表文件:")
    print("  - 图表1_模型准确率对比.png")
    print("  - 图表2_各类别准确率对比.png")
    print("  - 图表3_混淆矩阵热力图.png")
    print("  - 图表4_数据集分布.png")
    print("  - 图表5_分类评估指标.png")


# ========================= 主程序 =========================

def main():
    """主函数"""
    print("\n" + "#"*60)
    print("# 口罩检测 - 机器学习 + 深度学习完整项目")
    print("#"*60)
    
    # 步骤1：构建数据集
    success = build_classification_dataset()
    if not success:
        return
    
    # 步骤2和3：训练模型
    results = []
    results.append(train_svm())
    results.append(train_resnet18(num_epochs=5))
    
    # 步骤4：结果总结
    print_results_summary(results)
    
    # 混淆矩阵
    print("\n" + "="*60)
    print("混淆矩阵详情")
    print("="*60)
    
    for result in results:
        if result:
            plot_confusion_matrix(result['y_true'], result['y_pred'], result['name'])
    
    # 步骤5：生成可视化图表
    visualize_results(results)
    
    print("\n" + "#"*60)
    print("# ✓ 所有流程完成!")
    print("#"*60)
    print("\n生成的文件:")
    print(f"  - 数据集: {DATASET_DIR}/")
    print(f"  - 模型: {MODELS_DIR}/")
    print(f"  - 图表: 当前目录")
    print("\n下一步:")
    print("  1. 查看生成的数据集和模型")
    print("  2. 查看生成的5张统计图表")
    print("  3. 修改模型参数进行实验")
    print("  4. 尝试其他算法或数据集")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
