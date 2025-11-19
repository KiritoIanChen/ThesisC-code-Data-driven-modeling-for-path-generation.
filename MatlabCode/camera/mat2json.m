load('cameraParam.mat'); % 如果你保存过标定结果

K = cameraParams.IntrinsicMatrix'; % 注意 MATLAB 是列主序，需要转置

fx = K(1,1);
fy = K(2,2);
cx = K(1,3);
cy = K(2,3);

height = cameraParams.ImageSize(1);
width  = cameraParams.ImageSize(2);

% 深度相机没有标定出来的 scale，默认设置 1.0（或根据设备文档）
depth_scale = 1.0;

camera_info = struct( ...
    'cx', cx, ...
    'cy', cy, ...
    'fx', fx, ...
    'fy', fy, ...
    'height', height, ...
    'width', width, ...
    'depth_scale', depth_scale);

% 转换为 JSON 字符串
jsonText = jsonencode(camera_info);
disp(jsonText);

% 保存到文件
fid = fopen('camera.json','w');
fprintf(fid, '%s', jsonText);
fclose(fid);
