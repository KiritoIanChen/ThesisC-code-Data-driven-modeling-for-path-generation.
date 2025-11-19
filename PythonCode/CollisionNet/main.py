import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import joblib  # 用于加载 sklearn 保存的 scaler

from CollisionDetect import GJK  # 你的GJK函数模块
from CollisionDetect.GJKNN import CollisionNet
# 加载归一化参数
scaler = joblib.load("gjk_scaler.save")  # 之前用 sklearn 保存的 scaler

# 定义 shape 数据
shape1 = {
    'XData': np.array([-0.5, -0.5, -1.5, -1.5, -0.5, -0.5, -1.5, -1.5]),
    'YData': np.array([1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5]),
    'ZData': np.array([0, 0, 0, 0, 2, 2, 2, 2])
}
shape2 = {
    'XData': np.array([1, 1, -1, -1, 1, 1, -1, -1]) ,
    'YData': np.array([1, -1, 1, -1, 1, -1, 1, -1]) ,
    'ZData': np.array([0, 0, 0, 0, 2, 2, 2, 2])
}

# 合并点云数据，准备输入
obb1 = np.vstack([shape1['XData'], shape1['YData'], shape1['ZData']]).T.flatten()
obb2 = np.vstack([shape2['XData'], shape2['YData'], shape2['ZData']]).T.flatten()
raw_input = np.concatenate([obb1, obb2], axis=0).reshape(1, -1)  # 1x48

# 归一化输入
nn_input = scaler.transform(raw_input)
nn_input_tensor = torch.tensor(nn_input, dtype=torch.float32)

# 加载模型
model = torch.load("GJKNet.pt", weights_only=False)
model.eval()

# NN预测
start_time = time.time()
with torch.no_grad():
    output = model(nn_input_tensor)
    _, predicted = torch.max(output, 1)
    nn_result = predicted.item()
nn_time = time.time() - start_time

# GJK算法检测
start_time = time.time()
gjk_result = GJK.gjk(shape1, shape2, iterations=100)
gjk_time = time.time() - start_time

# 输出结果
print(f"Neural Net Prediction: {nn_result} (1=collision), Time: {nn_time:.6f} s")
print(f"GJK Result: {gjk_result} (1=collision), Time: {gjk_time:.6f} s")

# 可视化函数
def plot_obb(ax, shape, color='r'):
    points = np.vstack([shape['XData'], shape['YData'], shape['ZData']]).T

    # 立方体的12条边，点索引对
    edges = [
        [0,1], [1,3], [3,2], [2,0],  # 底面
        [4,5], [5,7], [7,6], [6,4],  # 顶面
        [0,4], [1,5], [2,6], [3,7]   # 侧边
    ]

    for edge in edges:
        xs, ys, zs = points[edge, 0], points[edge, 1], points[edge, 2]
        ax.plot(xs, ys, zs, c=color, linewidth=1.5)

    ax.scatter(points[:,0], points[:,1], points[:,2], c=color, s=10)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_obb(ax, shape1, color='blue')
plot_obb(ax, shape2, color='green')


# 绘制点云
points1 = np.vstack([shape1['XData'], shape1['YData'], shape1['ZData']]).T
points2 = np.vstack([shape2['XData'], shape2['YData'], shape2['ZData']]).T
ax.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c='blue', s=20, label='OBB1 Points')
ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c='green', s=20, label='OBB2 Points')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title(f"NN: {nn_result} ({nn_time:.4f}s) | GJK: {gjk_result} ({gjk_time:.4f}s)")
plt.show()

