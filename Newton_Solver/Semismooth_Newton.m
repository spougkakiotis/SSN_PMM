function [x,v,iter,tol_achieved,total_Krylov_iters,num_of_factor] = Semismooth_Newton(NS,tol,maxit,pl)
% =================================================================================================================================== %
% Semismooth_Newton: 
% ----------------------------------------------------------------------------------------------------------------------------------- %
% x = Semismooth_Newton(NS,tol,maxit,method,version,LRS)
%                                     takes as an input a MATLAB struct NS that containing relevant information needed 
%                                     to build the semismooth Newton system corresponding to minimizing the augmented 
%                                     Lagrangian with respect to x, or y(the primal or dual variable), the minimum accuracy 
%                                     tolerance and the maximum number of SSN iterations, the version of SSN-PMM used, as 
%                                     well as a structure for potential low-rank updates. It employs a semismooth 
%                                     Newton iteration (given the current iterate (x_k,y_k,z_k)) and returns the  
%                                     accepted "optimal" solution x, or y.
%
% Possible values for version:
%                               "Factorization" (uses factorization per SSN iteration),
%                               "Factorization-Preconditioner" (uses the same factorization for many SSN iterations
%                                                               correcting it with low-rank updates if needed),
%                               "Matrix-Free-Preconditioner" (employs a matrix-free preconditioner based on 
%                                                             a partial Cholesky decomposition),
%                               "Single-Factorization-Preconditioner" (uses the factorization computed during the 
%                                                                      search for a starting point).
% 
% Author: Spyridon Pougkakiotis.
% ___________________________________________________________________________________________________________________________________ %
    n = NS.n;
    m = NS.m;
    if (nargin < 4 || isempty(pl))    
        pl = 1;    
    end
    x = NS.x;               % Starting point for SSN -> x_0 = x_k.
    v = NS.v;
    % =============================================================================================================================== %
    % Set the semismooth Newton parameters
    % ------------------------------------------------------------------------------------------------------------------------------- %
    eta_1 = (1e-1)*tol;             % Maximum tolerance allowed when solving the corresponding linear systems.
    eta_2 = 0.1;                    % Determines the rate of convergence of SSN (that is, the rate is (1+eta_2)).
                                    % The trade-off: for larger eta_2, SSN is faster but CG is slower. (eta_2 in (0,1].)
    mu = (0.4995/2);                % Fraction of the decrease in Lagrangian predicted by linear extrapolation that we accept.
    delta = 0.995;                  % Maximum possible step-length used within the backtracking line-search.
    NS.nu = 1;                      % penalty used within backtracking linesearch
    % _______________________________________________________________________________________________________________________________ %
    
    % =============================================================================================================================== %
    % Initialize metric and preconditioning struct.
    % ------------------------------------------------------------------------------------------------------------------------------- %
    total_Krylov_iters = 0;         % count overall Krylov iterates
    iter = 0;                       % keep track of SSN iterations
    num_of_factor = 0;              % keep track of the number of factorizations performed
    max_outliers = 0;               % maximum number of outliers allowed before recomputing the preconditioner
    maxit_Krylov = 200;             % maximum number of Krylov iterations
    PS = struct();
    PS.n = n;    PS.m = m;
    dx = zeros(n,1);
    % _______________________________________________________________________________________________________________________________ %
 
    while (iter < maxit)
    % ------------------------------------------------------------------------------------------------------------------------------- %
    % SSN Main Loop structure:
    % Until (|| \nabla L(x_{k+1},y_{k+1},z_k) || <= tol) do
    %   Build matrices B_j, B_hat_j, belonging in the Clarke subdifferential of Proj_{K}(beta^{-1} z_k +  x_j),
    %   and in the Clarke subdifferential of prox_{zeta g}(x_j - zeta res_smooth_primal), respectively.
    %   Let matrix M_j = [-(Q + rho I + B_j) A^T; A beta^{-1} I] and compute an spd preconditioner if necessary
    %   Approximately solve the system: M_j d = - \nabla L(x_{k_j},y_{k_j},z_k), using minres
    %   Perform Line-search with Backtracking
    %   j = j + 1;
    % End
    % ------------------------------------------------------------------------------------------------------------------------------- %
        % =========================================================================================================================== %
        % Compute and store an element in the Clarke subdifferential of Proj_{K}(beta^{-1} z_k + x_j). 
        % --------------------------------------------------------------------------------------------------------------------------- %
        w = (1/NS.beta).*(NS.z) + x;  
        w_lb = (w <= NS.lb);
        w_ub = (w >= NS.ub);
        B = ones(n,1);
        B(w_lb) = zeros(nnz(w_lb),1);
        w(w_lb) = NS.lb(w_lb);
        B(w_ub) = zeros(nnz(w_ub),1);
        w(w_ub) = NS.ub(w_ub);
        NS.B = B;
        % ___________________________________________________________________________________________________________________________ %

        % =========================================================================================================================== %
        % Compute and store an element in the Clarke subdifferential of prox_{zeta g}(x_j - zeta res_smooth_primal). 
        % --------------------------------------------------------------------------------------------------------------------------- %
        res_smooth_primal = NS.c + NS.Q*x - NS.A_tr*v + NS.z + NS.beta.*x - NS.beta.*w + (1/NS.rho).*(x-NS.x);
        u_hat = x - NS.zeta.*res_smooth_primal;
        u_hat_active = ((abs(u_hat) > NS.zeta.*NS.D) | (NS.D == 0));
        B_hat = zeros(n,1);
        u_hat_size = nnz(u_hat_active);
        B_hat(u_hat_active) = ones(u_hat_size,1);
        u_hat_inactive = (B_hat == 0);
        % ___________________________________________________________________________________________________________________________ %


        
        
        % =========================================================================================================================== %
        % Compute the right-hand-side and check the termination criterion
        % --------------------------------------------------------------------------------------------------------------------------- %
        prox_u_hat = u_hat; % Proximity operator of g(x) = \|Dx\|_1.
        prox_u_hat = max(abs(prox_u_hat)-NS.zeta.*NS.D, zeros(n,1)).*sign(prox_u_hat);  
        rhs = [(1/NS.zeta).*(x - prox_u_hat);  % SSN right-hand-side.
               NS.b - NS.A*x - (1/NS.beta).*(v - NS.v)];
      
        res_error = norm(rhs);
        if (res_error < tol)     % Stop if the desired accuracy is reached.
            break;
        end

        iter = iter + 1;
        if (pl > 1)
            if (iter == 1)
                fprintf(NS.fid,'___________________________________________________________________________________________________\n');
                fprintf(NS.fid,'___________________________________Semismooth Newton method________________________________________\n');
            end
            fprintf(NS.fid,'SSN Iteration                                         Residual Infeasibility                    \n');
            fprintf(NS.fid,'%4d                                                      %9.2e                      \n',iter,res_error);
        end
        % ___________________________________________________________________________________________________________________________ %

        % =========================================================================================================================== %
        % Check if we need to re-compute the preconditioner. If so compute it and store its factors in PS
        % --------------------------------------------------------------------------------------------------------------------------- %
        if (iter == 1)
            update_preconditioner = true; % true if the preconditioner must be recomputed
        elseif ((nnz(PS.B - B) + nnz(PS.u_hat_active - u_hat_active) > max_outliers))
            update_preconditioner = true;
        else
            update_preconditioner = false;
        end
        PS.H_tilde = (NS.Q_diag + (NS.beta + (1/NS.rho)).*ones - NS.beta.*B); % update anyway!
        if (update_preconditioner)
            num_of_factor = num_of_factor + 1;
            PS.B = B;   PS.u_hat_active = u_hat_active; PS.u_hat_size = u_hat_size;
            NS.u_hat_active = u_hat_active; NS.u_hat_size = u_hat_size;
            NS.Q_bar = NS.Q(:,u_hat_active)';
            idx_set = ((B == 1) & u_hat_active);
            card_idx_set = nnz(idx_set);
            if (card_idx_set)
                Schur_tilde = NS.A(:,idx_set)*(spdiags((1./PS.H_tilde(idx_set)), 0, card_idx_set, card_idx_set)*...
                              NS.A(:,idx_set)') + (1/NS.beta).*speye(m);
            else
                Schur_tilde = speye(m);
            end
            [PS.L_S,chol_flag,PS.Perm] = chol(Schur_tilde,'lower','vector');     % Cholesky factorization
            if (chol_flag)
                fprintf("Numerical instability in the Cholesky decomposition of the preconditioner.\n");
                NS.beta = NS.beta*0.9;
                NS.rho = NS.rho*0.9;
                continue;
            end
            PS.PermInv(PS.Perm) = 1:m;
        end
        % ___________________________________________________________________________________________________________________________ %
        
        % =========================================================================================================================== %
        % Call MINRES using the previously built preconditioner to approximately solve the SSN sub-problem.
        % --------------------------------------------------------------------------------------------------------------------------- %
        dx(u_hat_inactive) = -rhs([u_hat_inactive;false(m,1)]);
        reduced_rhs = rhs([u_hat_active;true(m,1)]);
        reduced_rhs = reduced_rhs + ...
                      [NS.Q_bar(:,u_hat_inactive)*dx(u_hat_inactive);
                       -NS.A(:,u_hat_inactive)*dx(u_hat_inactive)];
        Krylov_tol = max(1e-8,min(min(eta_1,res_error^(1+eta_2)),1));
        [dx(u_hat_active),dv,Krylov_iter,instability,drop_direction] = ...
                                            Newton_Iterative_Solver(NS,reduced_rhs,PS,maxit_Krylov,Krylov_tol,pl);
        total_Krylov_iters = Krylov_iter + total_Krylov_iters;
 
        if (drop_direction || instability) % Decraese the penalty parameters, drop the direction and re-solve.
            NS.beta = NS.beta*0.9;
            NS.rho = NS.rho*0.9;
            continue;
        end
        % ___________________________________________________________________________________________________________________________ %
        if (iter == 1)
            alpha = 0.995;
        elseif (nnz(NS.D))
            alpha = Nonsmooth_Line_Search(NS,x,v,dx,dv,mu,delta);         
        else
            alpha = Backtracking_Line_Search(NS,x,v,dx,dv,mu,delta);
        end
        x = x + alpha.*dx;
        v = v + alpha.*dv;
    end
    tol_achieved = norm(rhs);
    if (pl > 1)
        fprintf(NS.fid,'___________________________________________________________________________________________________\n');
    end
end 
% *********************************************************************************************************************************** %
% END OF FILE.
% *********************************************************************************************************************************** %
