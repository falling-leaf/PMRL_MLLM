import torch
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def get_pooled_vectors(folder_path):
    """读取并压缩特征：[47, 10240] -> [10240]（使用 float64 提高精度）"""
    vectors = []
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.pt')])

    for f in file_names:
        file_path = os.path.join(folder_path, f)
        # 使用 float64
        data = torch.load(file_path, map_location='cpu').to(torch.float64)
        combined_vector = data.mean(dim=0)
        vectors.append(combined_vector)

    return torch.stack(vectors), file_names


def plot_similarity_heatmap(vectors, file_names, title, save_name):
    """计算余弦相似度并绘制热力图（数值稳定 + 正确对角线）"""

    # 1. 高精度归一化
    norm_vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)

    # 2. 余弦相似度矩阵
    sim_matrix = torch.mm(norm_vectors, norm_vectors.t())

    # 3. 数值修正（关键）
    sim_matrix = torch.clamp(sim_matrix, -1.0, 1.0)

    sim_np = sim_matrix.cpu().numpy()
    n = sim_np.shape[0]

    # 4. 统计信息（更高精度打印）
    off_diag = sim_np[~np.eye(n, dtype=bool)]
    avg_sim = off_diag.mean()

    print(
        f"{title}\n"
        f"  Average similarity (off-diagonal): {avg_sim:.8f}\n"
        f"  Min / Max: {off_diag.min():.8f} / {off_diag.max():.8f}"
    )

    # 5. 显式设置颜色范围（极其重要）
    vmin = off_diag.min()
    vmax = 1.0  # 对角线必须是视觉上的最大值

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        sim_np,
        cmap='YlGnBu',
        vmin=vmin,
        vmax=vmax,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={'label': 'Cosine Similarity'}
    )

    plt.title(f"{title}\nAverage Similarity (off-diagonal): {avg_sim:.4f}")
    plt.xlabel("Files Index")
    plt.ylabel("Files Index")

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()


# --- 执行 ---
path_a = "./tmp_base_samples"
path_b = "./tmp_rephrase_samples"

if os.path.exists(path_a):
    vecs_a, names_a = get_pooled_vectors(path_a)
    plot_similarity_heatmap(
        vecs_a, names_a,
        "Similarity: Base Samples",
        "sim_base.png"
    )

if os.path.exists(path_b):
    vecs_b, names_b = get_pooled_vectors(path_b)
    plot_similarity_heatmap(
        vecs_b, names_b,
        "Similarity: Rephrase Samples",
        "sim_rephrase.png"
    )
