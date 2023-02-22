function alpha = Nonsmooth_Line_Search(NS,x,v,dx,dv,mu,delta)
% ==================================================================================================================== %
% Nonsmooth Line Search
% -------------------------------------------------------------------------------------------------------------------- %
% alpha = Nonsmooth_Line_Search(NS,x,v,dx,dv,mu,delta)
%                                                    takes as an input the Newton structure (NS), containing all 
%                                                    relevant information needed to build the augmented Lagrangian, 
%                                                    x, which is the point upon which the Lagrangian is evaluated 
%                                                    (as well as its gradient), dx, which is the computed Newton 
%                                                    direction, mu, which is the fraction of decrease in the 
%                                                    Lagrangian, predicted by linear extrapolation, that we 
%                                                    are willing to accept, and finally, delta, which is the 
%                                                    maximum possible step-length.
%                                                    It returns the final step-length which satisfies the line-search
%                                                    requirements.
%                                                           
% Author: Spyridon Pougkakiotis.
% ____________________________________________________________________________________________________________________ %

    % ================================================================================================================ %
    % j-th SSN iteration: 
    % Compute and store phi(x_j) = c^T x_j + y_k^T(Ax_j) + (beta_k/2)||Ax_j - b||^2 -
    %                              x_j^T(Proj_{K^*}(z_k - beta_k x_j) + (1/(2rho_k))||x_j - x_k||^2
    %                              -(1/(2beta_k))||Proj_{K^*}(z_k - beta_k x_j) - z_k||^2
    % as well as \nabla (phi(x_j)) = c + A^Ty_k + beta_k A^T(Ax_j-b) - Proj_{K^*}(z_k - beta_k x_j) + (1/rho_k)(x_j-x_k)
    % ---------------------------------------------------------------------------------------------------------------- %
    w = (1/NS.beta).*(NS.z) + x;  
    w_lb = (w < NS.lb);
    w_ub = (w > NS.ub);
    w(w_lb) = NS.lb(w_lb);
    w(w_ub) = NS.ub(w_ub);
    res_smooth_primal = NS.c + NS.Q*x - NS.A_tr*v + NS.z + NS.beta.*x - NS.beta.*w + (1/NS.rho).*(x-NS.x);
  
    u_hat = x - NS.zeta.*res_smooth_primal;
    prox_u_hat = u_hat; % Proximity operator of g(x) = \|Dx\|_1.
    prox_u_hat = max(abs(prox_u_hat)-NS.zeta.*NS.D, zeros(NS.n,1)).*sign(prox_u_hat);  
    F_hat = [(1/NS.zeta).*(x - prox_u_hat);  
             NS.b - NS.A*x - (1/NS.beta).*(v - NS.v)];
    Theta = norm(F_hat,2)^2;
    % ________________________________________________________________________________________________________________ %
    
    % ================================================================================================================ %
    % Let alpha = delta, and evaluate phi(x + alpha.*dx).
    % ---------------------------------------------------------------------------------------------------------------- %
    alpha = delta;
    x_new = x + alpha.*dx;
    v_new = v + alpha.*dv;
    w = (1/NS.beta).*(NS.z) + x_new; 
    w_lb = (w < NS.lb);
    w_ub = (w > NS.ub);
    w(w_lb) = NS.lb(w_lb);
    w(w_ub) = NS.ub(w_ub);

    res_smooth_primal = NS.c + NS.Q*x_new - NS.A_tr*v_new + NS.z + NS.beta.*x_new - NS.beta.*w + (1/NS.rho).*(x_new-NS.x);


    u_hat = x_new - NS.zeta.*res_smooth_primal;
    prox_u_hat = u_hat; % Proximity operator of g(x) = \|Dx\|_1.
    prox_u_hat = max(abs(prox_u_hat)-NS.zeta.*NS.D, zeros(NS.n,1)).*sign(prox_u_hat);  
    F_hat_new = [(1/NS.zeta).*(x_new - prox_u_hat);  
                 NS.b - NS.A*x_new - (1/NS.beta).*(v_new - NS.v)];
    % ________________________________________________________________________________________________________________ %
    
    % ================================================================================================================ %
    % Iterate until you find a step-length satisfying the Armijo-Goldstein condition.
    % ---------------------------------------------------------------------------------------------------------------- %
    counter = 1;
    while (norm(F_hat_new)^2 > (1-2*mu*alpha)*Theta)
        counter = counter + 30;
        alpha = delta^counter;
        x_new = x + alpha.*dx;
        v_new = v + alpha.*dv;
        w = (1/NS.beta).*(NS.z) + x_new;   
        w_lb = (w < NS.lb);
        w_ub = (w > NS.ub);
        w(w_lb) = NS.lb(w_lb);
        w(w_ub) = NS.ub(w_ub);

        res_smooth_primal = NS.c + NS.Q*x_new - NS.A_tr*v_new + NS.z + NS.beta.*x_new - NS.beta.*w + (1/NS.rho).*(x_new-NS.x);
   
        u_hat = x_new - NS.zeta.*res_smooth_primal;
        prox_u_hat = u_hat; % Proximity operator of g(x) = \|Dx\|_1.
        prox_u_hat = max(abs(prox_u_hat)-NS.zeta.*NS.D, zeros(NS.n,1)).*sign(prox_u_hat);  
        F_hat_new = [(1/NS.zeta).*(x_new - prox_u_hat);  
                     NS.b - NS.A*x_new - (1/NS.beta).*(v_new - NS.v)];
    end
    % ________________________________________________________________________________________________________________ %
end
% ******************************************************************************************************************** %
% END OF FILE.
% ******************************************************************************************************************** %

