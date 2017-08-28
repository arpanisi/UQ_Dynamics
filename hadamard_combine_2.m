function [mu,Sigma] = hadamard_combine_2(mu_cell,Sigma_cell,Cl,Cl2)

% P : Association Matrix
% mu_cell: R (m x 1) x (1 x n)
% Sigma_cell: R(m x 1) x (n x n)

m = size(Cl,2); % Number of clusters
n = size(Cl,1); % Number of State variables


mu = zeros(n,1);
Sigma = zeros(n);

for i=1:m
    mu_j = zeros(n,1);
    Sigma_j = zeros(n);
    
    z_j = Cl(:,i);
    cl2 = Cl2(:,i);
    cl = logical(z_j);
    mu_cl_j = mu_cell{1,i};
    
    Sigma_cl_j = Sigma_cell{1,i};
    
    mu_j(cl,1) = mu_cl_j;
    mu_j(cl2,1) = 0;
    Sigma_j(cl,cl) = Sigma_cl_j;
    
    mu = mu + z_j.*mu_j;
    Sigma = Sigma + (z_j*z_j').*Sigma_j; 
end