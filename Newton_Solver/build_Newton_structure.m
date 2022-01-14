function NS = build_Newton_structure(Q,D,A,b,c,x,v,z,beta,rho,zeta,lb,ub,PMM_iter)
% ==================================================================================================================== %
% build_Newton_structure: Store all relevant information about the Newton system.
% -------------------------------------------------------------------------------------------------------------------- %
% NS = build_Newton_structure(A,A_tr,b,c,x,y,z,beta,rho,pos_varis,free_variables) returns a MATLAB 
%      struct that holds the relevant information of the Newton system, required for solving the step equations in
%      the SSN-PMM.
% 
% Author: Spyridon Pougkakiotis.
% ____________________________________________________________________________________________________________________ %
    % ================================================================================================================ %
    % Store all the relevant information required from the semismooth Newton's method.
    % ---------------------------------------------------------------------------------------------------------------- %
    m = size(v,1);
    n = size(x,1);
    NS = struct();
    NS.x = x;
    NS.v = v;
    NS.z = z;
    NS.b = b;
    NS.c = c;
    NS.m = m;
    NS.n = n;
    NS.beta = beta;
    NS.rho = rho;
    NS.zeta = zeta;
    NS.PMM_iter = PMM_iter;
    NS.lb = lb;
    NS.ub = ub;
    NS.A = A;
    NS.D = D;
    NS.Q = Q;
    NS.A_tr = A';
    NS.Q_diag = spdiags(Q,0);
    % ________________________________________________________________________________________________________________ %
end
% ******************************************************************************************************************** %
% END OF FILE.
% ******************************************************************************************************************** %
