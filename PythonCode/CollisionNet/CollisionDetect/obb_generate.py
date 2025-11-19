import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def obb_generate(coord_xyz, expand_dist):
    # 协方差矩阵和特征向量
    cov_matrix = np.cov(coord_xyz.T)
    eig_val, eig_vec = np.linalg.eig(cov_matrix)

    # 坐标变换到主方向空间
    coord_uvw = coord_xyz @ eig_vec

    # AABB 范围并拓展
    u_min, v_min, w_min = np.min(coord_uvw, axis=0) - expand_dist
    u_max, v_max, w_max = np.max(coord_uvw, axis=0) + expand_dist

    # AABB 八个顶点
    AABB_uvw = np.array([
        [u_min, v_min, w_min],
        [u_min, v_min, w_max],
        [u_min, v_max, w_min],
        [u_min, v_max, w_max],
        [u_max, v_min, w_min],
        [u_max, v_min, w_max],
        [u_max, v_max, w_min],
        [u_max, v_max, w_max]
    ])

    # 转换回 xyz 空间
    OBB_xyz = AABB_uvw @ eig_vec.T

    # 输出结构
    shape = {
        'XData': OBB_xyz[:, 0],
        'YData': OBB_xyz[:, 1],
        'ZData': OBB_xyz[:, 2]
    }

    return shape

def visualize_pointcloud_and_obb(coord_xyz, shape):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制原始点云
    ax.scatter(coord_xyz[:, 0], coord_xyz[:, 1], coord_xyz[:, 2], color='blue', s=2, label='Point Cloud')

    # OBB 顶点
    x, y, z = shape['XData'], shape['YData'], shape['ZData']
    obb_points = np.column_stack([x, y, z])

    # 定义 12 条棱边
    edges = [
        [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
        [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]
    ]
    for edge in edges:
        p1, p2 = obb_points[edge[0]], obb_points[edge[1]]
        ax.plot(*zip(p1, p2), color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('OBB Visualization')
    ax.legend()
    plt.tight_layout()
    plt.show()

# 示例测试
# if __name__ == "__main__":
#     # 随机生成一些 3D 点云
#     np.random.seed(0)
#     point_cloud = np.random.randn(100, 3) * 0.05 + np.array([1, 2, 3])
#
#     expand = 0.01
#     shape = obb_generate(point_cloud, expand)
#
#     visualize_pointcloud_and_obb(point_cloud, shape)


