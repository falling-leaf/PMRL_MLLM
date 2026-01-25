import torch
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_pooled_vectors(folder_path):
    """读取并压缩特征：[47, 10240] -> [10240]"""
    vectors = []
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.pt')])
    
    for f in file_names:
        file_path = os.path.join(folder_path, f)
        data = torch.load(file_path, map_location='cpu').to(torch.float32)
        # 平均池化降维
        combined_vector = torch.mean(data, dim=0) 
        vectors.append(combined_vector)
    
    return torch.stack(vectors), file_names

def plot_similarity_heatmap(vectors, file_names, title, save_name):
    """计算余弦相似度并绘制热力图"""
    # 1. 计算余弦相似度矩阵
    # 归一化向量
    norm_vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
    # 矩阵乘法得到相似度矩阵: [n, 10240] * [10240, n] -> [n, n]
    sim_matrix = torch.mm(norm_vectors, norm_vectors.t()).numpy()

    # 2. 统计信息
    avg_sim = (np.sum(sim_matrix) - len(vectors)) / (len(vectors) * (len(vectors) - 1))
    print(f"{title} - 平均相互相似度: {avg_sim:.4f}")

    # 3. 绘图
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        sim_matrix, 
        annot=False,  # 如果文件多，设为False以免数字重叠
        cmap='YlGnBu', # 颜色方案：黄色->绿色->蓝色
        xticklabels=False, # 隐藏坐标轴标签防止拥挤
        yticklabels=False,
        cbar_kws={'label': 'Cosine Similarity'}
    )
    
    plt.title(f"{title}\nAverage Similarity: {avg_sim:.4f}")
    plt.xlabel("Files Index")
    plt.ylabel("Files Index")
    
    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.show()

# --- 执行 ---
path_a = "./base_samples"
path_b = "./rephrase_samples"

# 文件夹 1
if os.path.exists(path_a):
    vecs_a, names_a = get_pooled_vectors(path_a)
    plot_similarity_heatmap(vecs_a, names_a, "Similarity: Base Samples", "sim_base.png")

# 文件夹 2
if os.path.exists(path_b):
    vecs_b, names_b = get_pooled_vectors(path_b)
    plot_similarity_heatmap(vecs_b, names_b, "Similarity: Rephrase Samples", "sim_rephrase.png")