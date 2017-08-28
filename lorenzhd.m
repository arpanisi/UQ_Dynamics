function dy = lorenzhd(X,e,w,sigma,rho,beta)
i = size(w,2);
n = size(w,1);
I = false(n+4,1);
if(i==1)
    I(4:n+3,1) = ones(n,1);
else
    I(4:n+3,1) = w(:,2);
end

num_states = sum(I);
ind_states = find(I(:,1)==1);

X0 = zeros(n+4,1);
X0(I==1) = X;

clearvars X
dy = zeros(num_states,1);
for ind=4:num_states+3
    i = ind_states(ind-3);
    rho0 = rho(i-3);
    sigma0 = sigma(i-3);
    beta0 = beta(i-3);
    if(mod(i,3)==1)
        dy(ind-3,1) = sigma0*(X0(i+1)-X0(i)) + e*(X0(i+3) - 2*X0(i) + X0(i-3));
    elseif(mod(i,3)==2)
        dy(ind-3,1) = X0(i-1)*(rho0 - X0(i+1)) - X0(i);
    else
        dy(ind-3,1) = X0(i-2)*X0(i-1)-beta0*X0(i);
    end
end

