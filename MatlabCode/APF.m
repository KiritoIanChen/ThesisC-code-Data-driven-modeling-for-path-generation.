% clear
% clc
% 
% StartPoint = [-300 300 400];
% GoalPoint = [300 350 800];
% thr = 5; % tolerance of end point
% 
% objRadius = 80;
% r = objRadius + 100; 
% 
% objMat = [-185 309 519;
%            103 349 568];
% objNum = size(objMat,1);


function path = APF(StartPoint,GoalPoint,obbinfo,r,thur)

    positionNow = StartPoint;
    
    k = 0.000001; 
    m = 200000;
    
    iterMax = 5000;
    iterStep = 5;
    iter = 1;
    gravitationForce = zeros(3,1);
    repulsiveForce = zeros(3,1);
    sumForce = zeros(3,1);
    angleGoal = zeros(3,1);
    obbNum = size(obbinfo,2);
    angleObj = [];
    
    path(1,:) = positionNow;
    
    % Main loop
    while iter < iterMax
        % 
        [angleObj, angleGoal] = calcuAngle(obbNum,obbinfo,positionNow,angleObj, GoalPoint);
        % gravititionForce
        gravitationForce = calcuGravitationForce(positionNow,GoalPoint,k,angleGoal);
        % repulsiveForce
        repulsiveForce = repulsiveForce*0;
        repulsiveForce = calcuRepulsiveForce(repulsiveForce,obbNum,positionNow,obbinfo,angleObj,r,m);
        % sumForce 
        sumForce = repulsiveForce + gravitationForce;
        % current location
        positionNow = positionNow + iterStep.*sumForce'./(sum(abs(sumForce)));
        iter = iter+1;
        path(iter,:) = positionNow';
        % break point
        if sqrt( (positionNow(1) - GoalPoint(1))^2 + (positionNow(2) - GoalPoint(2))^2 + (positionNow(3) - GoalPoint(3))^2 ) < thur
            break;
        end
    
    end

end

function [angleObj, angleGoal] = calcuAngle(obbNum,obbinfo,positionNow,angleObj,GoalPoint)
     
    for i = 1:obbNum

        pointNum = size(obbinfo(i).XData,2);

        for j = 1:pointNum
    
            deltaX = obbinfo(i).XData(j) - positionNow(1);
            deltaY = obbinfo(i).YData(j) - positionNow(2);
            deltaZ = obbinfo(i).ZData(j) - positionNow(3);
        
            r = sqrt(deltaX^2 + deltaY^2 + deltaZ^2);
            phi = atan2(deltaY,deltaX);
            theta = acos(deltaZ/r);
        
            x = sin(theta)*cos(phi);
            y = sin(theta)*sin(phi);
            z = cos(theta);
        
            angleObj(1,i) = x;
            angleObj(2,i) = y;
            angleObj(3,i) = z;

        end
    
    end
    
    deltaX = GoalPoint(1) - positionNow(1);
    deltaY = GoalPoint(2) - positionNow(2);
    deltaZ = GoalPoint(3) - positionNow(3);
    
    r = sqrt(deltaX^2 + deltaY^2 + deltaZ^2);
    phi = atan2(deltaY,deltaX);
    theta = acos(deltaZ/r);
    
    x = sin(theta)*cos(phi);
    y = sin(theta)*sin(phi); 
    z = cos(theta);
    
    angleGoal(1) = x;
    angleGoal(2) = y;
    angleGoal(3) = z;

end

function gravitationForce = calcuGravitationForce(positionNow,GoalPoint,k,angleGoal)

    r = sqrt( (positionNow(1) - GoalPoint(1))^2 + (positionNow(2) - GoalPoint(2))^2 + (positionNow(3) - GoalPoint(3))^2 );
    
    gravitationForce(1,1) = k*r*angleGoal(1);
    gravitationForce(2,1) = k*r*angleGoal(2); 
    gravitationForce(3,1) = k*r*angleGoal(3); 

end

function repulsiveForce = calcuRepulsiveForce(repulsiveForce,obbNum,positionNow,obbinfo,angleObj,r,m)
    counter = 1;
    for i = 1:obbNum

        pointNum = size(obbinfo(i).XData,2);

        for j = 1:pointNum

            p0 = sqrt( (positionNow(1) - obbinfo(i).XData(j))^2 + (positionNow(2) - obbinfo(i).YData(j))^2 + (positionNow(3) - obbinfo(i).ZData(j))^2 );
            
            if p0 > r(counter)
                repulsiveForce = repulsiveForce + 0;
            else 
                repulsiveForce = repulsiveForce - m*(1/p0 - 1/r(counter))*(1/p0^2)*angleObj(:,i);
            end
            counter = counter + 1;

        end
    
    end

end


