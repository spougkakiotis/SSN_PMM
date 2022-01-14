% Include Code and Data files.
curr_path = pwd;
addpath(genpath(curr_path)); 
fid = 1;   

fprintf('Should the default parameters be used?\n');
default = input('Type 1 for default parameters, or anything else to manually include them.\n');
if (default == 1)
    tol = 1e-5;                                             % Tolerance used.
    max_PMM_iter = 200;                                     % Maximum number of IP iterations.
    printlevel = 2;                                         % Printing choice (see IP-PMM documentation).
    problem_set = "Pearson_PDE_Optimization";
else
    fprintf('Choose a value for the allowed error tolerance.\n');
    while(true)
        tol = input('Type a double value in the form 1e-k, where k must be in [1,12].\n');
        if (isinf(tol) || isnan(tol) || ~isa(tol,'double') || tol > 1e0 || tol < 1e-12)
            fprintf('Incorrect input argument.\n');
        else
            break;
        end
    end
    fprintf('Choose the maximum number of PMM iterations.\n');
    while(true)
        max_PMM_iter = input('Type an integer value in k between [50,300].\n');
        if (isinf(max_PMM_iter) || isnan(max_PMM_iter) || floor(max_PMM_iter)~= max_PMM_iter || ...
            max_PMM_iter > 300 || max_PMM_iter < 50)
            fprintf('Incorrect input argument.\n');
        else
            break;
        end
    end
    fprintf('Choose the printlevel.\n');
    fprintf('                         0: no printing\n');
    fprintf('                         1: print PMM iterations and parameters\n');
    fprintf('                         2: also print SSN iterations\n');
    fprintf('                         3: also print Krylov iterations\n');
    while(true)
        printlevel = input('Type an integer value in k between [0,2].\n');
        if (isinf(printlevel) || isnan(printlevel) || ...
            floor(printlevel)~= printlevel || printlevel > 4 || printlevel < 1)
            fprintf('Incorrect input argument.\n');
        else
            break;
        end
    end
end




% User specification of the problem to be solved.
disp('Problem:');
disp('         1 - Poisson Control: L^1 + L^2-regularizer and bounded control.');
disp('         2 - Convection Diffusion: L^1 + L^2-regularizer and bounded control.');
problem_choice = input('Type 1 to 2 or anything else to exit.\n');
solution_statistics = Pearson_PDE_Test_Generator(problem_choice,tol,max_PMM_iter,printlevel,fid);

