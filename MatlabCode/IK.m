function q = IK(T_target, q_init, robot)

    max_steps = 5000;
    pos_tol = 1e-5;  % mm
    rot_tol = 1e-5;  % rad

    q = q_init;

    for i = 1:max_steps
        T_now = robot.fkine(q).T;

        % error
        e_pos = T_target(1:3,4) - T_now(1:3,4);

        R_now = T_now(1:3,1:3);
        R_goal = T_target(1:3,1:3);
        R_err = R_goal * R_now';
        e_rot = rotm2vec(R_err);  

        e = [e_pos; e_rot];

        if norm(e_pos) < pos_tol && norm(e_rot) < rot_tol
            return;
        end

        J = robot.jacob0(q);
        dq = pinv(J) * e;
        q = q + dq';
    end

    error('IK can not solve');
end