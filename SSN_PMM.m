function [solution_struct] = SSN_PMM(Q, D, A, b, c, lb, ub, tol, maxit, printlevel, print_fid)
% ======================================================================================================================== %
% This function is a primal-dual Semismooth Newton Proximal Method of Multipliers, suitable for solving convex quadratic 
% programming problems. The method takes as input a problem of the following form:
%
%                                    min   (1/2)(x)^T Q (x) + c^T x + \|Dx\|_1,
%                                    s.t.  A x = b,
%                                          lb <= x <= ub,
%
% where D is a diagonal positive semi-definite matrix, and solves it to epsilon-optimality, 
% returning the primal and dual optimal solutions (or a message indicating that the optimal solution was not found). 
%
% INPUT PARAMETERS:
% SSN_PMM(Q, D, A, b, c, lb, ub, tol, maxit, printlevel):  
%
%                            Q -> smooth Hessian matrix,
%                            D -> diagonal vector representing a positive semi-definite diagonal matrix,
%                            A -> constraint matrix, 
%                            b -> right-hand-side,
%                            c -> linear part of the objective,
%                            lb -> vector containing the lower-bound restrictions of x (unbounded default),
%                            ub -> vector containing the upper-bound restrictions of x (unbounded default),
%                            tol -> error tolerance (10^(-4) default),
%                            maxit -> maximum number of PMM iterations (200 default),
%                            printlevel -> 0 for not printing intermediate iterates,
%                                       -> 1 for printing only PMM iterates (default),
%                                       -> 2 for additionally printing SNN iterates.
%                                       -> 3 for also printing Krylov iterates.
%                            printf_fid -> file ID to print output.
%
% OUTPUT: [solution_struct], where, the struct contains the following entries
%         x: Optimal primal solution
%         v: Lagrange multiplier vector corresponding to equality constraints
%         z: Lagrange multiplier vector corresponding to inequalities
%         opt: 0, if the maximum number of iterations is reached,
%              1, if the tol-optimal solution was found,
%              2, if the method terminated due to numerical inaccuracy.
%         PMM_iter: number of PMM iterations to termination.
%         SSN_iter: number of SSN iterations to termination.
%         Krylov_iter: number of Krylov iterations to termination.
%         total_num_of_factorizations: the total number of Cholesky factorizations performed.
%
% Author: Spyridon Pougkakiotis, October 2021, Edinburgh.
% ________________________________________________________________________________________________________________________ %
    % ==================================================================================================================== %
    % Parameter filling and dimensionality testing.
    % -------------------------------------------------------------------------------------------------------------------- %
    if (~issparse(A))                    % Ensure that A is sparse.
        A = sparse(A);
    end
    if (~issparse(Q))                    % Ensure that Q is sparse.
        Q = sparse(Q);
    end
  
    if (issparse(b))  b = full(b);   end % Make sure that b, c are full.
    if (issparse(c))  c = full(c);   end

    % Make sure that b and c are column vectors of dimension m and n.
    if (size(b,2) > 1) b = (b)'; end
    if (size(c,2) > 1) c = (c)'; end
    m = size(b,1);  n = size(c,1);
    if (~isequal(size(c),[n,1]) || ~isequal(size(b),[m,1]) )
        error('problem dimensions incorrect');
    end
    if (isempty(D))
        D = zeros(n,1);
    elseif (size(D) == size(Q))
        D = spdiags(D,0);
    elseif (size(D) ~= size(c))
        error('Vector D representing the non-smooth Hessian is given incorrectly.'\n);
    end
    % Set default values for missing parameters.
    if (nargin < 6  || (isempty(lb)))           lb = -Inf.*ones(n,1);  end
    if (nargin < 7  || (isempty(ub)))           ub = Inf.*ones(n,1);   end
    if (nargin < 8  || isempty(tol))            tol = 1e-4;            end
    if (nargin < 9  || isempty(maxit))          maxit = 200;           end
    if (nargin < 10 || isempty(printlevel))     printlevel = 1;        end
    if (nargin < 11 || isempty(print_fid))      print_fid = 1;         end
    pl = printlevel;
    % ____________________________________________________________________________________________________________________ %
    
     A_tr = A';                           % Store the transpose for computational efficiency.

    % ==================================================================================================================== %
    % Initialization - Starting Point:
    % Choose an initial starting point (x,v,z) such that any positive variable is set to a positive constant and 
    % free variables are set to zero.
    % -------------------------------------------------------------------------------------------------------------------- %
    beta = 1e2;   rho = 5e2;   zeta = 1;                                % Initial primal and dual regularization values.
    warm_start_maxit = 100;
    warm_start_tol = 1e-3;
    [x,v,~,z,ws_opt] = SSN_PMM_warmstart(Q,D,A,A_tr,b,c,lb,ub,warm_start_tol,warm_start_maxit);
    % ____________________________________________________________________________________________________________________ %

    % ==================================================================================================================== %  
    % Initialize parameters
    % -------------------------------------------------------------------------------------------------------------------- %
    max_SSN_iters = 4000;                                               % Number of maximum SSN iterations. 
    PMM_iter = 0;                                                       % Monitors the number of PMM iterations.
    SSN_iter = 0;                                                       % Monitors the number of SSN iterations.
    total_num_of_factorizations = 0;                                    % Monitors the number of factorizations.
    Krylov_iter = 0;                                                    % Monitors the number of Krylov iterations.
    in_tol_thr = tol;                                                   % Inner tolerance for Semismooth Newton method.
    SSN_maxit = 8;                                                      % Maximum number of SSN iterations.
    opt = 0;                                                            % Variable monitoring the optimality.
    PMM_header(print_fid,pl);                                           % Set the printing choice.
    reg_limit = 1e+6;                                                   % Maximum value for the penalty parameters.
    solution_struct = struct();                                         % Struct to contain output information.
    [res_p,res_d,compl] = compute_residual(Q,D,A,A_tr,b,c,lb,ub,x,v,z);
    % ____________________________________________________________________________________________________________________ %
    

    while (PMM_iter < maxit)
    % -------------------------------------------------------------------------------------------------------------------- %
    % SSN-PMM Main Loop structure:
    % Until (||Ax_k - b|| < tol && ||x - prox_{g}(x-c - Qx + A^Tv_k - z_k)|| < tol && ||x - Proj_{K}(x+z)||) do
    %   Call Semismooth Newton's method to approximately minimize the primal-dual augmented Lagrangian w.r.t. x, y;
    %   update z;
    %   update the reuglarization paramters;
    %   k = k + 1;
    % End
    % -------------------------------------------------------------------------------------------------------------------- %
        % ================================================================================================================ %
        % Check termination criteria
        % ---------------------------------------------------------------------------------------------------------------- %
        if ((PMM_iter == 0)&& ws_opt && (warm_start_tol <= tol))
            fprintf('optimal solution found\n');
            opt = 1;
            break;
        elseif (norm(res_p,inf)/(1+norm(b)) < tol && norm(res_d,inf)/(1+norm(c)+norm(x,inf)) < tol ...
                &&  compl/(1 + norm(x,inf) + norm(z,inf)) < tol )
            fprintf('optimal solution found\n');
            opt = 1;
            break;
        end
%         elseif (norm(res_p,inf)/(1) < tol && norm(res_d,inf)/(1) < tol ...
%                 &&  compl/(1) < tol )
%             fprintf('optimal solution found\n');
%             opt = 1;
%             break;
%         end
% 
        PMM_iter = PMM_iter+1;        
        % ________________________________________________________________________________________________________________ %
                
        % ================================================================================================================ %
        % Build or update the Newton structure
        % ---------------------------------------------------------------------------------------------------------------- %
        if (PMM_iter == 1) 
            NS = build_Newton_structure(Q,D,A,b,c,x,v,z,beta,rho,zeta,lb,ub,PMM_iter);
            NS.fid = print_fid;
        else
            NS.x = x; NS.v = v; NS.z = z; NS.beta = beta; NS.rho = rho; NS.PMM_iter = PMM_iter; NS.zeta = zeta;
        end
        % ________________________________________________________________________________________________________________ %
        
        % ================================================================================================================ %
        % Call semismooth Newton method to find the x-update.
        % ---------------------------------------------------------------------------------------------------------------- %
        res_vec = [0.1*norm(res_p); 0.1*norm(res_d); 1];
        in_tol = max(min(res_vec),in_tol_thr);
        SSN_tol_achieved = 2*max(res_vec);
        counter = 0; 
        while (SSN_tol_achieved > max(1e-1*max(res_vec),min(res_vec))) % && counter < 1
            counter = counter + 1;
            [x, v, SSN_in_iters, SSN_tol_achieved, Krylov_in_iters, num_of_factorizations] = ...
                                 Semismooth_Newton(NS,in_tol,SSN_maxit,pl);
            Krylov_iter = Krylov_iter + Krylov_in_iters;
            total_num_of_factorizations = num_of_factorizations + total_num_of_factorizations;
            SSN_iter = SSN_iter + SSN_in_iters;
            NS.x = x;
            NS.v = v;
            if (SSN_iter >= max_SSN_iters)
                fprintf('Maximum number of inner iterations is reached. Terminating without optimality.\n');
                w = (1/beta).*(NS.z) + x;
                w_lb = find(w < lb);
                w_ub = find(w > ub);
                w(w_lb) = lb(w_lb);
                w(w_ub) = ub(w_ub);
                z = NS.z + beta.*x - beta.*w;
                break;
            end
        end      
        % ________________________________________________________________________________________________________________ %
        
        % ================================================================================================================ %
        % Perform the dual update (z).
        % ---------------------------------------------------------------------------------------------------------------- %
        w = (1/beta).*(NS.z) + x;
        w_lb = (w < lb);
        w_ub = (w > ub);
        w(w_lb) = lb(w_lb);
        w(w_ub) = ub(w_ub);
        z = NS.z + beta.*x - beta.*w;
        [new_res_p,new_res_d,compl] = compute_residual(Q,D,A,A_tr,b,c,lb,ub,x,v,z);
        % ________________________________________________________________________________________________________________ %
        
        % ================================================================================================================ %
        % If the overall primal and dual residual error is decreased, 
        % we increase the penalty parameters aggressively.
        % If not, we continue increasing the penalty parameters slower, limiting the increase to the value 
        % of the regularization threshold.
        % ---------------------------------------------------------------------------------------------------------------- %
        [beta,rho] = update_PMM_parameters(res_p,res_d,new_res_p,new_res_d,beta,rho,reg_limit);
        res_p = new_res_p;
        res_d = new_res_d;
        % ________________________________________________________________________________________________________________ %

        % ================================================================================================================ %
        % Print iteration output.  
        % ---------------------------------------------------------------------------------------------------------------- %
        pres_inf = norm(res_p,inf);
        dres_inf = norm(res_d,inf);  
        PMM_output(print_fid,pl,PMM_iter,SSN_iter,pres_inf,dres_inf,compl,SSN_tol_achieved,beta,rho)
        % ________________________________________________________________________________________________________________ %
    end % while (iter < maxit)
    % ==================================================================================================================== %  
    % The PMM has terminated. Print results, and prepare output.
    % -------------------------------------------------------------------------------------------------------------------- %
    [res_p,res_d,compl] = compute_residual(Q,D,A,A_tr,b,c,lb,ub,x,v,z);
    fprintf(print_fid,'outer iterations: %5d\n', PMM_iter);
    fprintf(print_fid,'inner iterations: %5d\n', SSN_iter);
    fprintf(print_fid,'primal feasibility: %8.2e\n', norm(res_p,inf));
    fprintf(print_fid,'dual feasibility: %8.2e\n', norm(res_d,inf));
    fprintf(print_fid,'complementarity: %8.2e\n', compl);  
    fprintf(print_fid,'total number of factorizations: %5d\n', total_num_of_factorizations);
    fprintf(print_fid,'total Krylov iterations: %5d\n', Krylov_iter);
    solution_struct.x = x;  solution_struct.v = v;  solution_struct.z = z;
    solution_struct.opt = opt;  solution_struct.PMM_iter = PMM_iter;
    solution_struct.SSN_iter = SSN_iter;    solution_struct.Krylov_iter = Krylov_iter;
    solution_struct.total_num_of_factorizations = total_num_of_factorizations;
    solution_struct.obj_val = c'*x + (1/2)*(x'*(Q*x)) + norm(D.*x,1);
    % ____________________________________________________________________________________________________________________ %
end
% ************************************************************************************************************************ %
% END OF FILE
% ************************************************************************************************************************ %