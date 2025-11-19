% % 打开相机
% addpath('C:\Program Files (x86)\Intel RealSense SDK 2.0\matlab\');
% import realsense.*
% clf();
% 
% % 初始化保存计数器
% imgCount = 1;
% % Initial figure
% figure(1);
% hFig = imshow(zeros(480,640,3,'uint8')); 
% title('Color Image');
% 
% % Make Pipeline object to manage streaming
% pipe = realsense.pipeline();
% 
% % Start streaming on an arbitrary camera with default settings
% profile = pipe.start();
% 
% % Main loop
% while ishandle(hFig)
% 
%     % Obtain frames from a streaming device
%     fs = pipe.wait_for_frames();
% 
%     color = fs.get_color_frame();
% 
%     data = color.get_data();
%     color_img = permute(reshape(data',[3,color.get_width(),color.get_height()]),[3 2 1]);
%    % 检查键盘输入
%     if hFig.CurrentCharacter == 'e'
%         filename = sprintf('img%d.jpg', imgCount);
%         imwrite(color_img, filename);
%         fprintf('Saved: %s\n', filename);
%         imgCount = imgCount + 1;
%         hFig.CurrentCharacter = ' ';  % 重置按键状态
%     elseif hFig.CurrentCharacter == 'q'
%         disp('Quitting...');
%         break;
%     end
% 
% end
% 
%  % Stop streaming
% pipe.stop();
    


% 添加 RealSense SDK 路径
addpath('C:\Program Files (x86)\Intel RealSense SDK 2.0\matlab\');
import realsense.*
clf();

% 初始化保存计数器
imgCount = 1;

% 创建图窗并设置按键回调函数
figHandle = figure(1);
set(figHandle, 'Name', 'RealSense D435 RGB Viewer - Press "e" to save, "q" to quit', ...
               'KeyPressFcn', @keypress_callback);

% 初始化显示图像
hImg = imshow(zeros(480,640,3,'uint8'));
title('Color Image');

% 初始化变量用于键盘输入
keyPressed = '';

% 创建 Pipeline 对象并启动
pipe = realsense.pipeline();
profile = pipe.start();

% 主循环
while ishandle(figHandle)
    % 获取帧
    fs = pipe.wait_for_frames();
    color = fs.get_color_frame();

    % 读取 RGB 图像
    data = color.get_data();
    color_img = permute(reshape(data',[3, color.get_width(), color.get_height()]), [3 2 1]);

    % 更新图像显示
    set(hImg, 'CData', color_img);
    drawnow;

    % 检查键盘输入
    if ~isempty(keyPressed)
        switch keyPressed
            case 'e'
                filename = sprintf('img%d.jpg', imgCount);
                imwrite(color_img, filename);
                fprintf('Saved: %s\n', filename);
                imgCount = imgCount + 1;

            case 'q'
                disp('Quitting...');
                break;
        end
        keyPressed = '';  % 重置
    end
end

% 停止相机
pipe.stop();
close(figHandle);

% 回调函数：按键记录到 keyPressed 变量
function keypress_callback(~, event)
    assignin('base', 'keyPressed', event.Key);
end
