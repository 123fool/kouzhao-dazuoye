# -*- coding: utf-8 -*-
"""
测试可视化功能 - 快速演示统计图表
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CLASS_NAMES = ["with_mask", "without_mask", "mask_incorrect"]

# 模拟结果数据
np.random.seed(42)
n_samples = 300

# 模型1 (SVM) 的结果
y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.4, 0.35, 0.25])
y_pred_svm = np.where(np.random.random(n_samples) < 0.82, y_true, np.random.choice([0, 1, 2], size=n_samples))

# 模型2 (ResNet18) 的结果
y_pred_resnet = np.where(np.random.random(n_samples) < 0.90, y_true, np.random.choice([0, 1, 2], size=n_samples))

# 计算准确率
acc_svm = np.sum(y_pred_svm == y_true) / len(y_true)
acc_resnet = np.sum(y_pred_resnet == y_true) / len(y_true)

results = [
    {"name": "SVM", "accuracy": acc_svm, "y_true": y_true, "y_pred": y_pred_svm},
    {"name": "ResNet18", "accuracy": acc_resnet, "y_true": y_true, "y_pred": y_pred_resnet}
]

print("\n" + "="*60)
print("生成可视化图表...")
print("="*60)

# ============ 图表1: 模型准确率对比 ============
fig1, ax1 = plt.subplots(figsize=(10, 6))

model_names = [r['name'] for r in results]
accuracies = [r['accuracy'] for r in results]
colors = ['#FF6B6B', '#4ECDC4']

bars = ax1.bar(model_names, accuracies, color=colors, 
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
fig2, ax2 = plt.subplots(figsize=(12, 6))

class_accuracies = {cls: [] for cls in CLASS_NAMES}

for result in results:
    y_true = result['y_true']
    y_pred = result['y_pred']
    
    for i, cls_name in enumerate(CLASS_NAMES):
        mask = y_true == i
        if np.sum(mask) > 0:
            acc = np.sum(y_pred[mask] == y_true[mask]) / np.sum(mask)
            class_accuracies[cls_name].append(acc)

x = np.arange(len(model_names))
width = 0.25

for i, cls_name in enumerate(CLASS_NAMES):
    offset = (i - 1) * width
    bars = ax2.bar(x + offset, class_accuracies[cls_name], width,
                  label=cls_name, alpha=0.8, edgecolor='black')
    
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
ax2.set_xticklabels(model_names)
ax2.legend(loc='lower right', fontsize=11)
ax2.set_ylim(0, 1.1)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('图表2_各类别准确率对比.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 图表2_各类别准确率对比.png")
plt.close()

# ============ 图表3: 混淆矩阵热力图 ============
fig3, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, result in enumerate(results):
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

# ============ 图表4: 数据集分布 ============
fig4, ax = plt.subplots(figsize=(10, 6))

# 模拟数据集分布数据
dataset_dist = {
    'with_mask': [280, 60, 45],
    'without_mask': [245, 52, 40],
    'mask_incorrect': [175, 38, 30]
}

splits = ['训练集', '验证集', '测试集']
x_pos = np.arange(len(splits))
width = 0.25
colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, (cls_name, counts) in enumerate(dataset_dist.items()):
    offset = (i - 1) * width
    bars = ax.bar(x_pos + offset, counts, width, label=cls_name, 
                  alpha=0.8, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=9)

ax.set_xlabel('数据集分割', fontsize=12, fontweight='bold')
ax.set_ylabel('样本数量', fontsize=12, fontweight='bold')
ax.set_title('数据集分布统计', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(splits)
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('图表4_数据集分布.png', dpi=300, bbox_inches='tight')
print("✓ 已保存: 图表4_数据集分布.png")
plt.close()

# ============ 图表5: 详细的分类报告 ============
fig5, ax5 = plt.subplots(figsize=(12, 6))

y_true = results[0]['y_true']
y_pred = results[0]['y_pred']
model_name = results[0]['name']

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
print("\n可以在当前目录中查看这些PNG图表文件")
