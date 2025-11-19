function shape = obbGenerate(coord_xyz,expand_dist)

    % 求协方差矩阵和特征向量
    covMatrix = cov(coord_xyz);
    [eigVector,~] = eig(covMatrix);

    % 坐标变换到主方向空间
    coord_uvw = coord_xyz * eigVector;

    % 求AABB的8个顶点（在uvw空间）   5
    uMin = min(coord_uvw(:,1));
    uMax = max(coord_uvw(:,1));
    vMin = min(coord_uvw(:,2));
    vMax = max(coord_uvw(:,2));
    wMin = min(coord_uvw(:,3));
    wMax = max(coord_uvw(:,3));
    % 拓展obb箱范围
    % expand_dist = 10;
    uMin = uMin - expand_dist;
    uMax = uMax + expand_dist;
    vMin = vMin - expand_dist;
    vMax = vMax + expand_dist;
    wMin = wMin - expand_dist;
    wMax = wMax + expand_dist;

    AABB_uvw = [
        uMin, vMin, wMin;
        uMin, vMin, wMax;
        uMin, vMax, wMin;
        uMin, vMax, wMax;
        uMax, vMin, wMin;
        uMax, vMin, wMax;
        uMax, vMax, wMin;
        uMax, vMax, wMax
    ];

    % 变换回xyz空间得到OBB的8个点
    OBB_xyz = AABB_uvw * eigVector';

    shape.XData = OBB_xyz(:,1);
    shape.YData = OBB_xyz(:,2);
    shape.ZData = OBB_xyz(:,3);

end