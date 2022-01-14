function [res_p,res_d,compl] = compute_residual(Q,D,A,A_tr,b,c,lb,ub,x,v,z)
% ==================================================================================================================== %
% This function takes the QP problem data as well as the current iterate as input, and outputs the 
% primal, dual infeasibilities, as well as complementarity.
% -------------------------------------------------------------------------------------------------------------------- %
    n = size(x,1);
    res_p = b - A*x;                                                    % Non-regularized primal residual
    temp_res_d = x - c - Q*x + A_tr*v -z;
    temp_res_d = max(abs(temp_res_d)-D, zeros(n,1)).*sign(temp_res_d);  % Proximity operator of g(x) = \|Dx\|_1.
    res_d = x - temp_res_d;                                             % Non-regularized dual residual.
    temp_compl = x + z;
    temp_lb = (temp_compl < lb);
    temp_ub = (temp_compl > ub);
    temp_compl(temp_lb) = lb(temp_lb);
    temp_compl(temp_ub) = ub(temp_ub);
    compl = norm(x -  temp_compl);                                      % Measure of the complementarity between x and z.
% ____________________________________________________________________________________________________________________ %
end
% ******************************************************************************************************************** %
% END OF FILE.
% ******************************************************************************************************************** %
