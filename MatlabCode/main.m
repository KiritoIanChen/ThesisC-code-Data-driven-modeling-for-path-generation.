clear
clc
clf
% Initial param
axisMin = [-710 -710 0];
axisMax = [-50 -50 700];
thur = 10;

% pathPoint = [0 0 0]
pathPoint = [-588.53, -133.30, 200; -250 -500 100];
Home = [0 deg2rad(-75) deg2rad(90) deg2rad(-105) deg2rad(-90) 0];
Goal = [0 deg2rad(-75) deg2rad(90) deg2rad(-105) deg2rad(-90) deg2rad(-45)];
% pathPoint = [0 deg2rad(-75) deg2rad(90) deg2rad(-105) deg2rad(-90) 0; 0 deg2rad(-75) deg2rad(90) deg2rad(-105) deg2rad(-90) deg2rad(-45)];

% Initial Robot
robot.alpha = [pi/2, 0, 0, pi/2, -pi/2, 0];
robot.d = [0.1807, 0, 0, 0.1742, 0.11985, 0.11655]*1000;
robot.a = [0, -0.4784, -0.36, 0, 0, 0]*1000;
dof = 6;

ur16e = InitiRobot(robot, dof);
% ur16e.fkine()
% 选择方法 1:RRT* 2:APF
ismethod = true;
load('GJKnet.mat','GJKnet','inputps');
% plot
obbs = drawObj(pathPoint,[-900 -900 0],[900 900 900]);

r = [];
for i = 1:length(obbs)
    shape = obbs(i);
    
    
    vertices = [shape.XData(:), shape.YData(:), shape.ZData(:)];  % 8×3
    
    
    maxDists = zeros(1, 8);
    
    for j = 1:8
        
        diffs = vertices - vertices(j, :);       % 8x3 
        dists = sqrt(sum(diffs.^2, 2));           % 8x1                              
        maxDists(j) = max(dists);
    end
    
    r = [r, maxDists];
end
        
    
% find path
totalRRTPath = [];
totalAPFPath = [];
RRTpath = [];
APFpath = [];
RRTtime = 0;
APFtime = 0;
for i = 1:size(pathPoint,1) - 1
    startPoint = pathPoint(i,:);
    goalPoint = pathPoint(i+1,:);
    tic;
    [RRTpath,T] = RRTStar(axisMin,axisMax,startPoint,goalPoint,obbs,thur,GJKnet,inputps);
    RRTtime = toc;
    tic;
    APFpath = APF(startPoint,goalPoint,obbs,r./2,50);
    APFtime = toc;
    if ~isempty(RRTpath)
        for j = 1:size(RRTpath,1) - 1
            line([RRTpath(j,1) RRTpath(j+1,1)],[RRTpath(j,2) RRTpath(j+1,2)],[RRTpath(j,3) RRTpath(j+1,3)],'Color','g','LineWidth',1);
        end
        totalRRTPath = [startPoint;totalRRTPath; RRTpath(2:end,:);goalPoint];
    end

    if ~isempty(APFpath)
        for j = 1:size(APFpath,1) - 1
            line([APFpath(j,1) APFpath(j+1,1)],[APFpath(j,2) APFpath(j+1,2)],[APFpath(j,3) APFpath(j+1,3)],'Color','b','LineWidth',1);
        end
        totalAPFPath = [startPoint;totalAPFPath; APFpath(2:end,:);goalPoint];
    end

end
% legend({'RRTPath','APFPath'},'Location','southwest');

RRTStartLen = 0;
for i = 1:size(totalRRTPath,1) - 1
    RRTStartLen = RRTStartLen + CalcuDistance(totalRRTPath(i,:), totalRRTPath(i+1,:));
end

APFLen = 0;
for i = 1:size(totalAPFPath,1) - 1
    APFLen = APFLen + CalcuDistance(totalAPFPath(i,:), totalAPFPath(i+1,:));
end

disp(['Length of RRT*-NN path:' num2str(RRTStartLen)]);
disp(['Time cost of RRT*-NN:' num2str(RRTtime)]);
disp(['Length of APF path:' num2str(APFLen)]);
disp(['Time cost of APF:' num2str(APFtime)]);

Tt = eye(4);
R = rpy2r([0, pi, pi/2]);
Tt(1:3,1:3) = R;
qs = [];
datas = [];
isok = false;
qPrevious = [0 deg2rad(-75) deg2rad(90) deg2rad(-105) deg2rad(-90) 0];

if ismethod
    totalPath = totalRRTPath;
else 
    totalPath = totalAPFPath;
end

%% 解析解
% for i = 1:size(totalPath,1)
%     T = 
% end
% ur16e.plot(qs);

% save('TrainData.mat','datas');
%% 神经网络逆解
% load('net.mat','net','inputps', 'outputps');
% for i = 1:size(totalPath,1)
%     if i > 1
%         Tt(1:3,4) = totalPath(i,:);
%         data = [Tt(1:3,1)' Tt(1:3,2)' Tt(1:3,3)' Tt(1:3,4)' robot.alpha robot.a robot.d qPrevious];  % 1×36
%         % === 归一化输入 ===
%         data_norm = mapminmax('apply', data', inputps);  % 注意转置
% 
%         % === 网络推理 ===
%         output_norm = sim(net, data_norm);
% 
%         % === 反归一化输出 ===
%         output = mapminmax('reverse', output_norm, outputps);  % 反归一化
% 
%         qPrevious = output';
%         qs = [qs; qPrevious];
%     else
%         qs = [qs; qPrevious];
%     end
% end
% 
% ur16e.plot(qs);


%% URSIM
% % % TCP Host and Port settings
% host = '127.0.0.1'; % THIS IP ADDRESS MUST BE USED FOR THE VIRTUAL BOX VM
% % host = '192.168.230.128'; % THIS IP ADDRESS MUST BE USED FOR THE VMWARE
% % host = '192.168.0.100'; % THIS IP ADDRESS MUST BE USED FOR THE REAL ROBOT
% port = 30003;
% 
% 
% % Calling the constructor of rtde to setup tcp connction
% rtde = rtde(host,port);
% traj = totalPath;
% % Setting home
% home = [-588.53, -133.30, 371.91, 2.2214, -2.2214, 0.00];
% 
% % poses = rtde.movej(home);
% % Creating a path array
% path = [];
% 
% % setting move parameters
% v = 0.5;
% a = 1.2;
% blend = 0.005;
% 
% % Populate the path array
% for i = 1:length(traj)
%     %disp(i);
%     %disp(traj(i,1:3) + [-588.53, -133.30 100]);
%     point = [ [traj(i,1:3), (home(4:6))] ,a,v,0,blend];
%     if isempty(path)
%         path = point;
%     else
%         path = cat(1,path,point);
%     end
% end
% 
% % Execute the movement!
% pose1 = rtde.movej(path);
% pause(0.5);
% %pose2 = rtde.movej(home);
% % pose2 = [];
% % poses = [pose1; pose2];
% 
% rtde.drawPath(pose1);
% disp('Program completed');
% rtde.close;



function dis = CalcuDistance(A,B)

dis = sqrt((A(1) - B(1))^2 + (A(2) - B(2))^2 + (A(3) - B(3))^2);

end


function shapes = drawObj(pathPoint,axisMin,axisMax)

figure(1);
hold on;
% scatter3(pathPoint(1:end-1,1),pathPoint(1:end-1,2),pathPoint(1:end-1,3),'MarkerEdgeColor','k','MarkerFaceColor','r');
% scatter3(pathPoint(end,1),pathPoint(end,2),pathPoint(end,3),'MarkerEdgeColor','k','MarkerFaceColor','b');
shapes = [];
% Ape
len = 100/2;
width = 100/2;
high = 110;
shape.XData = [len, len, -len, -len, len, len, -len, -len];
shape.YData = [width, -width, width, -width, width, -width, width, -width];
shape.ZData = [0,0,0,0,high,high,high,high];
center = [-302.5,-405,55]';
R = rotated(0, 0, -pi/4);
shape= tf(shape,center,R);
shapes = [shapes,shape];

% Cat
len = 150/2;
width = 80/2;
high = 140;
shape.XData = [len, len, -len, -len, len, len, -len, -len];
shape.YData = [width, -width, width, -width, width, -width, width, -width];
shape.ZData = [0,0,0,0,high,high,high,high];
center = [-302.5,-195,70]';
R = rotated(0, 0, 0);
shape = tf(shape,center,R);
% shapes = [shapes,shape];

% Duck
len = 120/2;
width = 90/2;
high = 110;
shape.XData = [len, len, -len, -len, len, len, -len, -len];
shape.YData = [width, -width, width, -width, width, -width, width, -width];
shape.ZData = [0,0,0,0,high,high,high,high];
center = [-597.5,-195,55]';
R = rotated(0, 0, pi/2+pi/4);
shape= tf(shape,center,R);
shapes = [shapes,shape];

% egg box
len = 170/2;
width = 120/2;
high = 90;
shape.XData = [len, len, -len, -len, len, len, -len, -len];
shape.YData = [width, -width, width, -width, width, -width, width, -width];
shape.ZData = [0,0,0,0,high,high,high,high];
center = [-597.5,-405,45]';
R = rotated(0, 0, 0);
shape= tf(shape,center,R);
shapes = [shapes,shape];

% water bottle
len = 200/2;
width = 120/2;
high = 215;
shape.XData = [len, len, -len, -len, len, len, -len, -len];
shape.YData = [width, -width, width, -width, width, -width, width, -width];
shape.ZData = [0,0,0,0,high,high,high,high];
center = [-450,-300,107.5]';
R = rotated(0, 0, pi/4);
shape= tf(shape,center,R);
shapes = [shapes,shape];


% OBB的12条棱连接顺序
edges = [1 2; 1 3; 1 5; 2 4; 2 6; 3 4; 3 7; ...
         4 8; 5 6; 5 7; 6 8; 7 8];

for i = 1:length(shapes)

    obb.XData  = shapes(i).XData;
    obb.YData  = shapes(i).YData;
    obb.ZData  = shapes(i).ZData;

    OBB1_points = [obb.XData; obb.YData; obb.ZData]';

    % 画OBB1的12条棱（红色）
    for i = 1:size(edges,1)
        pt1 = OBB1_points(edges(i,1),:);
        pt2 = OBB1_points(edges(i,2),:);
        plot3([pt1(1), pt2(1)], [pt1(2), pt2(2)], [pt1(3), pt2(3)], 'r-', 'LineWidth', 2);
    end

end

view(3);
grid on;
axis equal;
axis([axisMin(1) axisMax(1) axisMin(2) axisMax(2) axisMin(3) axisMax(3)]);
xlabel('x');
ylabel('y');
zlabel('z');
shapes = shapes';

end


function irobot = InitiRobot(robot, dof)
    % DH Link
    for i = 1:dof
        L(i) = Link('d', robot.d(i), 'a', robot.a(i), 'alpha', robot.alpha(i));
        L(i).qlim = [-2*pi,2*pi];
    end

    % UR16E
    irobot = SerialLink(L, 'name', 'UR16E');
    irobot.plotopt={'jointdiam',0.001};
    
end

function T = FK(theta, alpha, d, a)

    A1 = GetT(theta(1,1), alpha(1), d(1), a(1));
    A2 = GetT(theta(1,2), alpha(2), d(2), a(2));
    A3 = GetT(theta(1,3), alpha(3), d(3), a(3));
    A4 = GetT(theta(1,4), alpha(4), d(4), a(4));
    A5 = GetT(theta(1,5), alpha(5), d(5), a(5));
    A6 = GetT(theta(1,6), alpha(6), d(6), a(6));
    T  = A1 * A2 * A3 * A4 * A5 * A6;
end

function T = GetT(theta, alpha, d, a)

    ca = cos(alpha);
    sa = sin(alpha);
    ct = cos(theta);
    st = sin(theta);

    T = [
        ct, -st * ca,  st * sa,  a * ct;
        st,  ct * ca, -ct * sa,  a * st;
        0,    sa,      ca,       d;
        0,    0,       0,        1
    ];

end

function shape = tf(shape,T,R)
    points = [shape.XData; shape.YData; shape.ZData];
    mpoints = R*points + T;
    shape.XData = mpoints(1,:);
    shape.YData = mpoints(2,:);
    shape.ZData = mpoints(3,:);
end

function R = rotated(rx, ry, rz)
% rotationMatrix3D 生成三维旋转矩阵
% 输入：
%   rx, ry, rz —— 绕 X、Y、Z 轴的旋转角度（单位：弧度）
% 输出：
%   R —— 3x3 旋转矩阵
%
% 旋转顺序为 Rz * Ry * Rx

    % 绕 X 轴旋转
    Rx = [1      0           0;
          0  cos(rx)   -sin(rx);
          0  sin(rx)    cos(rx)];

    % 绕 Y 轴旋转
    Ry = [cos(ry)   0   sin(ry);
          0         1       0;
         -sin(ry)   0   cos(ry)];

    % 绕 Z 轴旋转
    Rz = [cos(rz)  -sin(rz)   0;
          sin(rz)   cos(rz)   0;
          0              0     1];

    % 最终旋转矩阵（Z-Y-X 顺序）
    R = Rz * Ry * Rx;
end