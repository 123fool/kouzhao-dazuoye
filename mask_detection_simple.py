# -*- coding: utf-8 -*-
"""
口罩检测课程作业 - 简化版本（不依赖numpy的全局导入）
包含：
1. 数据集构建（从XML标注构建分类数据集）
2. ResNet18模型训练（深度学习）
3. 完整的性能评估

使用说明：
1. 修改 ROOT 变量为你的数据路径
2. 运行：python mask_detection_simple.py
"""

import os
import sys
import shutil
import random
import xml.etree.ElementTree as ET
import time
from collections import defaultdict
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
    print("❌ 错误: PyTorch 不可用")
    sys.exit(1)

# ========================= 配置部分 =========================

ROOT = r"D:\网盘\面部口罩检测数据集"  # 改为你的数据路径

IMG_DIR = os.path.join(ROOT, "images")
ANN_DIR = os.path.join(ROOT, "annotations")
DATASET_DIR = "dataset"
MODELS_DIR = "models"

# 数据集配置
random.seed(42)
torch.manual_seed(42)
CLASS_NAMES = ["with_mask", "without_mask", "mask_incorrect"]

print("=" * 60)
print("🎯 口罩检测 - 深度学习版本")
print("=" * 60)

# ========================= 第1部分：数据集构建 =========================

def parse_annotation(xml_file):
    """解析VOC格式XML文件"""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        filename = root.find('filename').text
        objects = []
        
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            objects.append((name, (xmin, ymin, xmax, ymax)))
        
        return filename, objects
    except Exception as e:
        print(f"解析 {xml_file} 失败: {e}")
        return None, []

def build_classification_dataset():
    """构建分类数据集"""
    print("\n📁 正在构建分类数据集...")
    
    # 检查原始数据
    if not os.path.exists(IMG_DIR):
        print(f"❌ 图像目录不存在: {IMG_DIR}")
        return False
    
    if not os.path.exists(ANN_DIR):
        print(f"❌ 标注目录不存在: {ANN_DIR}")
        return False
    
    xml_files = [f for f in os.listdir(ANN_DIR) if f.endswith('.xml')]
    print(f"   找到 {len(xml_files)} 个标注文件")
    
    # 创建数据集目录
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR)
    
    # 按类别创建目录
    data_by_split = defaultdict(lambda: defaultdict(list))
    
    for split in ['train', 'val', 'test']:
        for class_name in CLASS_NAMES:
            os.makedirs(os.path.join(DATASET_DIR, split, class_name), exist_ok=True)
    
    # 处理所有XML文件
    processed = 0
    skipped = 0
    
    for xml_file in xml_files:
        xml_path = os.path.join(ANN_DIR, xml_file)
        filename, objects = parse_annotation(xml_path)
        
        if filename is None or not objects:
            skipped += 1
            continue
        
        img_path = os.path.join(IMG_DIR, filename)
        if not os.path.exists(img_path):
            skipped += 1
            continue
        
        try:
            img = Image.open(img_path)
            
            for obj_class, (xmin, ymin, xmax, ymax) in objects:
                if obj_class not in CLASS_NAMES:
                    continue
                
                # 裁剪人脸区域
                face_img = img.crop((xmin, ymin, xmax, ymax))
                
                # 随机分配到train/val/test
                rand = random.random()
                if rand < 0.7:
                    split = 'train'
                elif rand < 0.85:
                    split = 'val'
                else:
                    split = 'test'
                
                # 保存图像
                save_dir = os.path.join(DATASET_DIR, split, obj_class)
                save_path = os.path.join(save_dir, f"{processed}_{obj_class}.jpg")
                face_img.save(save_path)
                data_by_split[split][obj_class].append(save_path)
                
            processed += 1
            
            if processed % 100 == 0:
                print(f"   已处理 {processed} 张图像...")
        
        except Exception as e:
            skipped += 1
    
    print(f"\n✅ 数据集构建完成！")
    print(f"   - 已处理: {processed} 张")
    print(f"   - 跳过: {skipped} 张")
    
    # 打印统计信息
    for split in ['train', 'val', 'test']:
        print(f"\n   {split.upper()}集:")
        total = 0
        for class_name in CLASS_NAMES:
            count = len(data_by_split[split][class_name])
            print(f"      {class_name}: {count}")
            total += count
        print(f"      总计: {total}")
    
    return True

# ========================= 第2部分：深度学习模型 =========================

def train_resnet18(num_epochs=5):
    """训练ResNet18模型"""
    print("\n🤖 正在训练ResNet18模型（深度学习）...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   使用设备: {device}")
    
    # 数据增强和转换
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # 加载数据集
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(DATASET_DIR, 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(DATASET_DIR, 'val'), data_transforms['val']),
        'test': datasets.ImageFolder(os.path.join(DATASET_DIR, 'test'), data_transforms['test'])
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=0),
        'val': DataLoader(image_datasets['val'], batch_size=32, shuffle=False, num_workers=0),
        'test': DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=0)
    }
    
    # 加载预训练的ResNet18
    model = models.resnet18(pretrained=True)
    
    # 冻结前面的层
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    
    # 修改最后的全连接层
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))
    
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f"\n   Epoch {epoch + 1}/{num_epochs}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(dataloaders['train'])
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(dataloaders['val'])
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"      Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"      Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 测试阶段
    print(f"\n   正在评估测试集...")
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    import numpy as np
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    test_acc = test_correct / test_total
    
    # 保存模型
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'resnet18_mask.pth')
    torch.save(model.state_dict(), model_path)
    print(f"   ✅ 模型已保存到: {model_path}")
    
    # 计算每个类别的准确率
    print(f"\n   📊 测试集结果:")
    print(f"      总体准确率: {test_acc:.4f}")
    
    for i, class_name in enumerate(CLASS_NAMES):
        mask = all_labels == i
        if np.sum(mask) > 0:
            class_acc = np.sum(all_preds[mask] == all_labels[mask]) / np.sum(mask)
            print(f"      {class_name}: {class_acc:.4f}")
    
    return {
        'model': 'ResNet18',
        'accuracy': test_acc,
        'predictions': all_preds,
        'labels': all_labels
    }

def main():
    """主函数"""
    start_time = time.time()
    
    # 第1步：构建数据集
    if not build_classification_dataset():
        print("❌ 数据集构建失败")
        return
    
    # 第2步：训练深度学习模型
    if not TORCH_AVAILABLE:
        print("❌ PyTorch不可用")
        return
    
    results = []
    results.append(train_resnet18(num_epochs=5))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("✅ 所有任务完成！")
    print("=" * 60)
    
    elapsed = time.time() - start_time
    print(f"⏱️  总耗时: {elapsed:.2f} 秒")

if __name__ == "__main__":
    main()
