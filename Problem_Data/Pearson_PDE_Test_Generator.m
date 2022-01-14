function [solution_statistics] = Pearson_PDE_Test_Generator(problem_choice,tol,max_PMM_iter,printlevel,fid)
% ====================================================================================================== %
% This function takes an integer as an input, which specifies the problem 
% that is being solved by SSN-PMM. Then, it generates and return the relevant data 
% required to solve the problem, in the form of a structure.
% ------------------------------------------------------------------------------------------------------ %
% Input
%        problem_choice: 1 -> Poisson Control with L^1/L^2-regularization and control bounds     
%                        2 -> Convection Diffusion with L^1/L^2-regularization and state and control bounds
%                        
% Output
%        solution_statistics: A structure of structures containing all relevant problem data.
% ______________________________________________________________________________________________________ %

    % ================================================================================================== %
    % Build the struct that will contain all solution info.
    % -------------------------------------------------------------------------------------------------- %
    solution_statistics = struct();
    solution_statistics.total_PMM_iters = 0; solution_statistics.total_time = 0;
    solution_statistics.total_SSN_iters = 0;
    solution_statistics.problem_converged = 0; solution_statistics.tol = tol;
    solution_statistics.total_Krylov_iters = 0;
    solution_statistics.objective_value = 0;    % To keep objective values.
    solution_statistics.status = 0;             % To keep convergence status.
    solution_statistics.total_num_of_factorizations = 0;   % To keep the maximum nnz of a Cholesky factor found.
    % __________________________________________________________________________________________________ %

    
    grid_type = 1; % uniform grid
    % ================================================================================================== %
    % Request input parameters from the user or use the default ones.
    % -------------------------------------------------------------------------------------------------- %
    fprintf('Should the default parameters be used?\n');
    default = input('Type 1 for default parameters, or anything else to manually include them.\n');
    if (default == 1)
            nc = 8;
            alpha_2 = 0;
            alpha_1 = 1e-3;    % regularization parameter of the L1 norm
    else 
        fprintf('Choose the number of uniform grid points.\n');
        while(true)
            nc = input('Type an integer k in [2,15] to enforce 2^k+1 grid points in each direction.\n');
            if (isinf(nc) || isnan(nc) || floor(nc)~= nc || nc >= 16 || nc <= 1)
                fprintf('Incorrect input argument. Please type an integer k between 3 and 15.\n');
            else
                break;
            end
        end
        fprintf('Choose a value for the smooth regularization parameter.\n');
        while(true)
            alpha_2 = input('Type a double value in the form 1e-k, where k must be in [0,12].\n');
            if (isinf(alpha_2) || isnan(alpha_2) || ~isa(alpha_2,'double') ||...
                alpha_2 > 1e-0 || alpha_2 < 1e-12)
                fprintf('Incorrect input argument.\n');
            else
                break;
            end
        end
        if (problem_choice == 1 || problem_choice == 2)
            fprintf('Choose a value for the L1 regularization parameter\n');
            while(true)
                alpha_1 = input('Type a double value in the form 1e-k, where k must be in [0,12].\n');
                if (isinf(alpha_1) || isnan(alpha_1) || ~isa(alpha_1,'double') ||...
                    alpha_1 > 1e-0 || alpha_1 < 1e-12)
                    fprintf('Incorrect input argument.\n');
                else
                    break;
                end
            end
        end
        if (problem_choice == 1)
            fprintf('Choose two uniform lower and upper bounds for the control.\n');
            while (true)
                u_alpha = input('Type the lower bound as a real number.\n');
                u_beta = input('Type the upper bound as a real number.\n');
                if (u_alpha > u_beta)
                    fprintf('Incorrect input argument.\n');
                else
                    break;
                end
            end
        end
        if (problem_choice == 4)
           fprintf('Choose two uniform lower and upper bounds for the state and the control.\n');
            while (true)
                u_alpha = input('Type the lower bound as a real number.\n');
                u_beta = input('Type the upper bound as a real number.\n');
                if (u_alpha >= u_beta)
                    fprintf('Incorrect input argument.\n');
                    continue;
                end
                y_alpha = input('Type the lower bound as a real number.\n');
                y_beta = input('Type the upper bound as a real number.\n');
                if (y_alpha >= y_beta)
                    fprintf('Incorrect input argument.\n');
                else
                    break;
                end
            end   
            fprintf('Choose a real value for the diffusion coefficient.\n');
            while (true)
                epsilon = input('Type the value as a real number in [0.01,0.5].\n');
                if (epsilon < 0.01 || epsilon > 0.5)
                    fprintf('Incorrect input argument.\n');
                else
                    break;
                end
            end
        end
    end
    % __________________________________________________________________________________________________ %
     fileID = fopen('./Output_files/Pearson_PDE_tabular_format_PMM_runs.txt','a+');

     % How large is resulting stiffness or mass matrix?
     np = (2^nc+1)^2; % entire system is thus 3*np-by-3*np

     % Compute matrices specifying location of nodes
    [x_1,x_2,x_1x_2,bound,mv,mbound] = square_domain_x(nc,grid_type);


    if (problem_choice == 1)
        problem_name = "Bounded_Poisson_L1_Control";
        if (default == 1)
            u_alpha = -2; % control is constrained to be within range [-alpha,alpha]
            u_beta = 1.5;
        end

        O = sparse(np,np);

        % Compute connectivity, stiffness and mass matrices (D and J)
        [ev,ebound] = q1grid(x_1x_2,mv,bound,mbound);
        [D,J_y] = femq1_diff(x_1x_2,ev);
        R = (sum(J_y))'; % equivalently diag(sum(J)), as J is symmetric
        R(bound) = 0;  % account for the boundary conditions
        
        % Specify vectors relating to desired state, source term and Dirichlet BCs
        yhat_vec = sin(pi*x_1x_2(:,1)).*sin(pi*x_1x_2(:,2));
        bc_nodes = ones(length(bound),1);

        % Initialize RHS vector corresponding to desired state
        Jyhat = J_y*yhat_vec;

        % Enforce Dirichlet BCs on state, and zero Dirichlet BCs on adjoint
        [D,b] = nonzerobc_input(D,zeros(np,1),x_1x_2,bound,bc_nodes);
        [J_y,Jyhat] = nonzerobc_input(J_y,Jyhat,x_1x_2,bound,bc_nodes);
        J_constr = J_y; for i = 1:length(bound), J_constr(bound(i),bound(i)) = 0; end
        obj_const_term = (1/2).*(yhat_vec'*Jyhat);      % Constant term included in the objective.
        c = [-Jyhat; zeros(np,1)];
    
        J_u = alpha_2.*J_y;
        A = [D -J_constr];
        Q = [J_y     O ;
             O      J_u];
        L1_D = [zeros(np,1);alpha_1.*R];
        lb = -Inf.*ones(2*np,1);
        ub = Inf.*ones(2*np,1);
        lb(np+1:2*np) = u_alpha.*ones(np,1);
        ub(np+1:2*np) = u_beta.*ones(np,1);
        [A,L1_D,Q,c,b,lb,ub] = Problem_scaling_set_up(A,L1_D,Q,c,b,lb,ub,2*np,np);
        
        tStart = tic;
        [solution_struct] = SSN_PMM(Q, L1_D, A, b, c, lb, ub, tol, max_PMM_iter, printlevel, fid);
        tEnd = toc(tStart);      
    elseif (problem_choice == 2)
        problem_name = "Conv_Diff_L1_Control";
        if (default == 1)
            u_alpha = -2; % control is constrained to be within range [-alpha,alpha]
            u_beta = 1.5;
            y_alpha = -inf;  y_beta = inf;       % Upper and lower bounds for state and control.
            epsilon = 0.05; % diffusion coefficient
        end

        O = sparse(np,np);

        % Compute connectivity, stiffness and mass matrices (D and J)
        [ev,ebound] = q1grid(x_1x_2,mv,bound,mbound);
        [K,N,J_y,epe,eph,epw] = femq1_cd(x_1x_2,ev);

        % Compute SUPG stabilization matrix
        epe = epe/epsilon;
        esupg = find(epe <= 1); expe = epe;
        if any(expe)
           supg = inf;
           if isinf(supg)
              expe = 0.5*(1-1./expe);
              expe(esupg) = inf;
           else
              expe = ones(size(expe)); expe = supg*expe; expe(esupg) = inf;
           end
           epp = expe; epp(esupg) = 0; epp = epp.*eph./epw;
           S = femq1_cd_supg(x_1x_2,ev,expe,eph,epw);
        end

        % Compute relevant matrices for optimization algorithm
        D = epsilon*K+N+S;
        
        R = (sum(J_y))'; % equivalently diag(sum(J)), as J is symmetric
        R(bound) = 0;  % account for the boundary conditions


        % Specify vectors relating to desired state and Dirichlet BCs
        yhat_vec = exp(-64.*((x_1x_2(:,1)-0.5).^2 + (x_1x_2(:,2)-0.5).^2));
        bc_nodes = 0.*x_1x_2(bound,1);
 
        % Initialize RHS vector corresponding to desired state
        Jyhat = J_y*yhat_vec;

        % Enforce Dirichlet BCs on state, and zero Dirichlet BCs on adjoint
        [D,b] = nonzerobc_input(D,zeros(np,1),x_1x_2,bound,bc_nodes);
        [J_y,Jyhat] = nonzerobc_input(J_y,Jyhat,x_1x_2,bound,bc_nodes);
        J_constr = J_y; for i = 1:length(bound), J_constr(bound(i),bound(i)) = 0; end
        obj_const_term = (1/2).*(yhat_vec'*Jyhat);      % Constant term included in the objective.
        c = [-Jyhat; zeros(np,1)];
    
        J_u = alpha_2.*J_y;
       % J_u = alpha_2.*(J_y+K);
        A = [D -J_constr];
        Q = [J_y     O;
             O      J_u];
         
        L1_D = [zeros(np,1);alpha_1.*R];
        lb = y_alpha.*ones(2*np,1);
        ub = y_beta.*ones(2*np,1);
        lb(np+1:2*np) = u_alpha.*ones(np,1);
        ub(np+1:2*np) = u_beta.*ones(np,1);

        [A,L1_D,Q,c,b,lb,ub] = Problem_scaling_set_up(A,L1_D,Q,c,b,lb,ub,2*np,np);

        tStart = tic;
        [solution_struct] = SSN_PMM(Q, L1_D, A, b, c, lb, ub, tol, max_PMM_iter, printlevel, fid);
        tEnd = toc(tStart);
    end
    % ================================================================================================== %
    % Gather and print all solution info.
    % -------------------------------------------------------------------------------------------------- %    
    solution_statistics.total_PMM_iters = solution_statistics.total_PMM_iters + solution_struct.PMM_iter;
    solution_statistics.total_time = solution_statistics.total_time + tEnd;  
    solution_statistics.total_SSN_iters = solution_statistics.total_SSN_iters + solution_struct.SSN_iter;
    solution_statistics.total_Krylov_iters = solution_statistics.total_Krylov_iters + solution_struct.Krylov_iter;
    solution_statistics.total_num_of_factorizations = solution_struct.total_num_of_factorizations;
    solution_statistics.objective_value = solution_struct.obj_val + obj_const_term;
    solution_statistics.status = solution_struct.opt;
    solution_statistics.y = solution_struct.x(1:np);
    solution_statistics.u = solution_struct.x(np+1:2*np,1);
    solution_statistics.v = solution_struct.v;
    solution_statistics.z = solution_struct.z;
 
    if (solution_struct.opt == 1)                                       % Success
       solution_statistics.problem_converged = 1;
       fprintf(fileID,'The optimal solution objective is %d.\n',solution_struct.obj_val);
    elseif (solution_struct.opt == 2)                                   % Primal Infeasibility
        fprintf(fileID,['Primal infeasibility has been detected.\n',...
                        'Returning the last iterate.\n']); 
    elseif (solution_struct.opt == 3)                                   % Dual Infeasibility
        fprintf(fileID,['Dual infeasibility has been detected\n',...
                        'Returning the last iterate.\n']); 
    elseif (solution_struct.opt == 4)                                   % Numerical Error
        fprintf(fileID,['Method terminated due to numerical error.\n',...   
                        'Returning the last iterate.\n']); 
    else                                                                % Reached maximum iterations
       fprintf(fileID,['Maximum number of iterations reached.\n',...
                           'Returning the last iterate.\n']); 
    end
    fprintf(fileID,['Name = %s & PMM iters = %d  & SSN iters = %d & Krylov iters = %d  & ' ...
        'Time = %.2e & opt = %s  \n'],problem_name, solution_struct.PMM_iter, ...
                                      solution_struct.SSN_iter, solution_struct.Krylov_iter,...
                                      solution_statistics.total_time, string(solution_struct.opt == 1)); 
    % __________________________________________________________________________________________________ %
                                  
end

