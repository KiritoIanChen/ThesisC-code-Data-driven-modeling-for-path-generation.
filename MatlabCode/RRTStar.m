function [path,T] = RRTStar(axisMin,axisMax,startPoint,goalPoint,obbinfo,thur,GJKnet,inputps)

iterMax = 50000;
iter = 0;
step = 5;
count = 1;
% thur = 10;
randProbability = 0.8;
r = 5*step;
flag = 0;

T.x(1) = startPoint(1);
T.y(1) = startPoint(2);
T.z(1) = startPoint(3);
T.pre(1) = 0;
T.cost(1) = 0;
path = [];

% main loop
while iter <= iterMax
    iter = iter + 1;
    % generate random point
    randCoor = randsample(axisMin,axisMax,goalPoint,randProbability);
    % find the nearest point
    [nearestCoor,parentIndex] = FindNearestPoint(T,randCoor);
    % get new point
    newCoor = ExpandPoint(nearestCoor,randCoor,step);
    % rewrite parent point
    parentIndex = RewriteFunction(T,newCoor,r,parentIndex);
    % detect collision
    A = [T.x(parentIndex),T.y(parentIndex),T.z(parentIndex)];
    B = newCoor;
    collisionFlag = collisionDetection(obbinfo,A,B,GJKnet, inputps);

    if collisionFlag
        continue;
    end
    % random son
    count = count + 1;
    T.x(count) = newCoor(1);
    T.y(count) = newCoor(2);
    T.z(count) = newCoor(3);
    T.pre(count) = parentIndex;
    T.cost(count) = CalcuDistance(A,B) + T.cost(parentIndex);

    % line([A(1) B(1)],[A(2) B(2)],[A(3) B(3)],'lineWidth',1);
    % pause(0.01);

    % random relink
    T = RandRelink(T,newCoor,obbinfo,r,GJKnet, inputps);

    if CalcuDistance(newCoor,goalPoint) < thur
        flag = 1;
        % find the nearest parent
        allNodes = [T.x' T.y' T.z'];
        dGoal = sqrt(sum((allNodes - goalPoint).^2,2));
        [~, idxGoal] = min(dGoal);
        % get the best path
        path = [];
        while idxGoal ~= 0
            path = [allNodes(idxGoal,:); path];
            idxGoal = T.pre(idxGoal);
        end

        break;
    end

end

if ~flag
    disp('Did not find the path');
    return;
else
    disp('success');
end

end

%%TRY
function randCoor = randsample(axisMin,axisMax,goalPoint,randProbability)
    if rand < randProbability
        % 在指定立方体范围内均匀采样
        % randCoor = axisMin + (axisMax - axisMin).*rand(1,3);
        randCoor(1,1) = (axisMax(1)-axisMin(1))*rand() + axisMin(1);
        randCoor(1,2) = (axisMax(2)-axisMin(2))*rand() + axisMin(2);
        randCoor(1,3) = (axisMax(3)-axisMin(3))*rand() + axisMin(3); 
    else
        % 以较小概率直接采样目标点，增强收敛
        randCoor = goalPoint;
    end
end

function [nearestCoor,parentIndex] = FindNearestPoint(T,randCoor)
    nodeList = [T.x' T.y' T.z'];
    distances = sqrt(sum((nodeList - randCoor).^2,2));
    [~,parentIndex] = min(distances);
    nearestCoor = nodeList(parentIndex,:);
end

function newCoor = ExpandPoint(nearestCoor,randCoor,step)
    v = randCoor - nearestCoor;
    d = norm(v);
    if d <= step
        newCoor = randCoor;
    else
        newCoor = nearestCoor + v * (step / d);
    end
end

function parentIndex = RewriteFunction(T,newCoor,r,parentIndex)
    % 在r半径范围内寻找更优父节点
    nodeList = [T.x' T.y' T.z'];
    distances = sqrt(sum((nodeList - newCoor).^2,2));
    candidateIdx = find(distances < r);

    minCost = T.cost(parentIndex) + norm(newCoor - nodeList(parentIndex,:));
    for i = 1:length(candidateIdx)
        idx = candidateIdx(i);
        cost = T.cost(idx) + norm(newCoor - nodeList(idx,:));
        if cost < minCost
            parentIndex = idx;
            minCost = cost;
        end
    end
end

function collisionFlag = collisionDetection(obbinfo,A,B,GJKnet, inputps)

    safetyMargin = 10;
    collisionFlag = false;

    % 中点和主方向向量
    mid = (A + B) / 2;
    dir = (B - A);
    dir = dir / norm(dir);

    % 构建中点四个方向的点（横向/垂向扩展）
    % 选择两个垂直于dir的方向
    temp = [1, 0, 0];
    if abs(dot(temp, dir)) > 0.99
        temp = [0, 1, 0]; % 防止共线
    end
    v1 = cross(dir, temp); v1 = v1 / norm(v1);
    v2 = cross(dir, v1);  % 第三个正交向量

    % 安全边距控制扩展
    p1 = mid + v1 * safetyMargin;
    p2 = mid - v1 * safetyMargin;
    p3 = mid + v2 * safetyMargin;
    p4 = mid - v2 * safetyMargin;

    % 点集构成: A, B, mid, 四个方向点
    points = [A; B; mid; p1; p2; p3; p4];  % 7x3
    shape = obbGenerate(points,5);

    for i = 1:length(obbinfo)
        obb.XData  = obbinfo(i).XData';
        obb.YData  = obbinfo(i).YData';
        obb.ZData  = obbinfo(i).ZData';

        collisionFlag = GJK(shape,obb,100);
        % collisionFlag = GJKNN(shape,obbinfo(i), GJKnet, inputps);
        % disp(collisionFlag);
        if collisionFlag
            break;
        end
    end

    
end

function T = RandRelink(T,newCoor,obbinfo,r,GJKnet, inputps)
    nodeList = [T.x' T.y' T.z'];
    distances = sqrt(sum((nodeList - newCoor).^2,2));
    neighborIdx = find(distances < r);
    currIndex = length(T.x);   % 新加入节点为最后一个

    for i = 1:length(neighborIdx)
        idx = neighborIdx(i);
        pt = nodeList(idx,:);
        % 如果通过newCoor连接到idx更优，且无碰撞，则更新
        if T.cost(currIndex) + norm(newCoor - pt) < T.cost(idx)
            % 碰撞检测（由newCoor到pt）
            if ~collisionDetection(obbinfo,newCoor,pt,GJKnet, inputps)
                T.pre(idx)  = currIndex;
                T.cost(idx) = T.cost(currIndex) + norm(newCoor - pt);
            end
        end
    end
end

function dis = CalcuDistance(A,B)

dis = sqrt((A(1) - B(1))^2 + (A(2) - B(2))^2 + (A(3) - B(3))^2);

end

function flag = GJKNN(shape1, shape2, GJKnet, inputps)

    % === 拼接输入数据 ===
    data = [shape1.XData(:)', shape1.YData(:)', shape1.ZData(:)', ...
            shape2.XData(:)', shape2.YData(:)', shape2.ZData(:)']';  % 48×1 列向量

    % === 手动归一化 ===
    xmin = inputps.xmin(:);
    xmax = inputps.xmax(:);
    data_norm = (data - xmin) ./ (xmax - xmin + eps);  % 归一化到 [0,1]

    % === 网络推理 ===
    output_norm = GJKnet(data_norm);

    flag = output_norm >= 0.7;  % 如果碰撞概率 >= 0.7，则认为是碰撞

end

% function flag = GJKNN(shape1, shape2, GJKnet, xmin, xmax, ymin, ymax)
%     % === 拼接输入数据 ===
%     data = [shape1.XData(:)', shape1.YData(:)', shape1.ZData(:)', ...
%             shape2.XData(:)', shape2.YData(:)', shape2.ZData(:)']';  % 48×1 列向量
%     data_norm = (data - xmin) ./ (xmax - xmin + eps);
%     output_norm = GJKnet(data_norm);
%     simu = output_norm .* (ymax - ymin + eps) + ymin;
%     flag = simu >= 0.5;
% end