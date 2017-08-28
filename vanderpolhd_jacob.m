function dy = vanderpolhd_jacob(X,e,mu)
dim = length(X);

x = X(1:2:dim);
y = X(2:2:dim);
mu = mu(1:2:dim);
n = length(x);

clearvars X
X(1) = x(n);
X(2:n+1,1) = x;
X(n+2) = x(1);

Y(2:n+1,1) = y;

for i=2:n+1
       mu0 = mu(i-1);
       dy(2*(i-1)+1,2*i+1) = e;
       dy(2*(i-1)+1,2*(i+1)+1) = e;
       dy(2*(i-1)+1,2*(i-1)+1) = mu0*(1-X(i)^2) - 2*e;
       dy(2*(i-1)+1,2*(i-1)+2) = -mu0*Y(i);
       dy(2*(i-1)+2,2*(i-1)+1) = 1/mu0;
       dy(2*(i-1)+2,2*(i-1)+2) = 0;
end

dy = dy(3:2*(n+1),3:2*(n+1));