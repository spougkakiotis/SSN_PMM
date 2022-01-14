function [x,y,y_2,z,opt] = SSN_PMM_warmstart(Q,D,A,A_tr,b,c,lb,ub,tol,maxit)
% ======================================================================================================================== %
% SSN PMM Warm-start:
% ------------------------------------------------------------------------------------------------------------------------ %
% [x,y,z] = SSN_PMM_warmstart(c,A,b,lb,ub,beta,rho,tol,maxit)
%                                           takes as an input the QP problem data, and applies
%                                           proximal ADMM to find an approximate solution
%                                           of the following primal-dual problem:
%
%            min_{x,w}        c^T x +(1/2)x^T Q x + g(w) + delta_{K}(w),                                    (P)
%            s.t.             Ax = b,   w = x,
%
%            max_{x,y,z}      y_1^T b - (1/2)x^T Q x - delta_K^*(z) - g^*(A^Ty - c - Qx - z),               (D)
%
%                                           where g(x) = \|Dx\|_1,
%                                           with tolerance tol. It terminates after maxit iterations.
%                                           It returns an approximate primal-dual solution (x,y,z).
%                                                           
% Author: Spyridon Pougkakiotis.
% ________________________________________________________________________________________________________________________ %
    % ==================================================================================================================== %
    % Initialize parameters and relevant statistics.
    % -------------------------------------------------------------------------------------------------------------------- %
    [m,n] = size(A);
    sigma = 1e-1;                                                       % Penalty parameter of the proximal ADMM.
    gamma = 1.618;                                                      % ADMM step-length.
    x = zeros(n,1); z = zeros(n,1); y = zeros(m,1); w = zeros(n,1);     % Starting point for ADMM.
    iter = 0;   opt = 0;
    % ____________________________________________________________________________________________________________________ %
    
    % ==================================================================================================================== %
    % Compute residuals (for termination criteria) and factorize the coefficient matrix of the main pADMM sub-problem.
    % -------------------------------------------------------------------------------------------------------------------- %
    temp_compl = w + z;
    temp_compl = max(abs(temp_compl)-D, zeros(n,1)).*sign(temp_compl);  % Proximity operator of g(x) = \|Dx\|_1.
    temp_lb = (temp_compl < lb);
    temp_ub = (temp_compl > ub);
    temp_compl(temp_lb) = lb(temp_lb);                                  % Euclidean projection to K.
    temp_compl(temp_ub) = ub(temp_ub);
    compl = norm(w -  temp_compl);                                      % Measure of the complementarity between w and z.
    res_p = [b-A*x; w-x];                                               % Primal residual
    res_d = c+Q*x-A_tr*y+z;                                             % Dual residual. 
    Q_tilde = gamma.*(max(norm(Q,1),1e-4) + spdiags(Q,0));              % We implicitly use regularizer R_x = \|Q\|_1 I_n - Off(Q).
    M = [spdiags(-Q_tilde,0,n,n)                 A_tr                                  -speye(n);
          A                     spdiags((1/(sigma*gamma)).*ones(n,1),0,m,m)            sparse(m,n);
          -speye(n)                             sparse(n,m)                 spdiags((1/(sigma*gamma)).*ones(n,1),0,n,n)];   
    [L,D_L,pp] = ldl(M,'lower','vector');
    % ____________________________________________________________________________________________________________________ %

    while(iter < maxit)
        % ================================================================================================================ %
        % Check termination criteria.
        % ---------------------------------------------------------------------------------------------------------------- %
        if (norm(res_p)/(1+norm(b)) < tol && norm(res_d)/(1+norm(c)) < tol &&  compl/(1 + norm(w) + norm(z)) < tol )
            fprintf('optimal solution found\n');
            opt = 1;
            break;
        end
        iter = iter+1;
        % ________________________________________________________________________________________________________________ %
        
        % ================================================================================================================ %
        % 1st sub-problem: calculation of w_{j+1} (prox evaluation of g() and then projection to K). 
        % ---------------------------------------------------------------------------------------------------------------- %
        w = x + (1/sigma).*z;
        w = max(abs(w)-(1/sigma).*D, zeros(n,1)).*sign(w);
        w_lb = (w < lb);
        w_ub = (w > ub);
        w(w_lb) = lb(w_lb);
        w(w_ub) = ub(w_ub);
        % ________________________________________________________________________________________________________________ %
        
        % ================================================================================================================ %
        % 2nd sub-problem: calculation of y_{j+1}.
        % ---------------------------------------------------------------------------------------------------------------- %
        rhs = [gamma.*(c + Q*x) - Q_tilde.*x + (1-gamma).*(A_tr*y - z);
               b + (1/(gamma*sigma)).*y;
               -w + (1/(gamma*sigma)).*z];  
        warn_stat = warning;
        warning('off','all');
        lhs = L'\(D_L\(L\(rhs(pp))));
        warning(warn_stat);
        lhs(pp) = lhs;
        x = lhs(1:n);   y = lhs(n+1:n+m);   z = lhs(n+m+1:end); 
        % ________________________________________________________________________________________________________________ %
        
        
        % ================================================================================================================ %
        % Residual Calculation.
        % ---------------------------------------------------------------------------------------------------------------- %       
        temp_compl = w + z;
        temp_compl = max(abs(temp_compl)-D, zeros(n,1)).*sign(temp_compl);
        temp_lb = (temp_compl < lb);
        temp_ub = (temp_compl > ub);
        temp_compl(temp_lb) = lb(temp_lb);
        temp_compl(temp_ub) = ub(temp_ub);
        compl = norm(w -  temp_compl);                                  % Measure of the complementarity between w and z.
        res_p = [b-A*x; w-x];                                           % Primal residual
        res_d = c+Q*x-A_tr*y+z;                                         % Dual residual.
        % ________________________________________________________________________________________________________________ %
    end
    fprintf('ADMM iterations: %5d\n', iter);
    fprintf('primal  feasibility: %8.2e\n', norm(res_p));
    fprintf('dual feasibility: %8.2e\n', norm(res_d));
    fprintf('complementarity: %8.2e\n', compl); 
    y_2 = z;
    z = retrieve_reformulated_z(D,w,y_2);
end


