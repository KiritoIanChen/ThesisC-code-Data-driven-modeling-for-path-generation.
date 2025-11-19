%% 1. 初始化
clear
close all
clc
format short

%% 2. 读取数据
load("OBB_GJK_Data_49xN.mat", "datas");
disp(size(datas));
datas = datas(randperm(size(datas,1)), :); % 打乱顺序

input = datas(:,1:48);      % 48维输入
output = datas(:,49);       % 二分类输出（0或1）

N = size(output,1);         
testNum = 15000;
trainNum = N - testNum;

%% 3. 训练集 & 测试集
input_train = input(1:trainNum,:)';
output_train = output(1:trainNum,:)';
input_test = input(trainNum+1:end,:)';
output_test = output(trainNum+1:end,:)';

%% 4. 数据归一化
[inputn, inputps] = mapminmax(input_train, 0, 1);
inputn_test = mapminmax('apply', input_test, inputps);

% 二分类用 one-hot 编码 [1; 0] 表示类别0， [0; 1] 表示类别1
outputn = full(ind2vec(output_train + 1)); 
outputn_test = full(ind2vec(output_test + 1));

%% 5. 构建神经网络：48输入 -> 17隐藏 -> 2输出 + softmax
GJKnet = patternnet(17);  % 保持和原BP一致的结构
GJKnet.performFcn = 'crossentropy';  % 交叉熵损失函数
GJKnet.trainFcn = 'trainlm';         % 快速训练函数
GJKnet.divideParam.trainRatio = 0.9;
GJKnet.divideParam.valRatio = 0.1;
GJKnet.divideParam.testRatio = 0;
GJKnet.trainParam.showWindow = false;  % 关闭GUI窗口

%% 6. 训练网络
[GJKnet, tr] = train(GJKnet, inputn, outputn);

%% 7. 测试输出 + 评估
output_pred = GJKnet(inputn_test);  % 2×N softmax输出
[~, pred_class] = max(output_pred); % 预测类别
[~, true_class] = max(outputn_test); % 真实类别

accuracy = sum(pred_class == true_class) / length(true_class);
fprintf("Test accuracy: %.4f\n", accuracy);

%% === 混淆矩阵绘图 ===
figure;
plotconfusion(outputn_test, output_pred);
title(sprintf("Confusion Matrix (Accuracy = %.2f%%)", accuracy*100));

%% === 绘制 ROC 曲线（可选）===
figure;
plotroc(outputn_test, output_pred);
title("ROC Curve");

%% 8. 保存模型
model_path = sprintf("GJKnet.mat");
save(model_path, "GJKnet", "inputps");
fprintf("Model saved to %s\n", model_path);


