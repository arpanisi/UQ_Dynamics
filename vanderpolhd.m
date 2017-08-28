function dy = vanderpolhd(X,u,e,w,mu)

i = size(w,2);
n = size(w,1);
I = false(n+3,1);
if(i==1)
    I(3:n+2,1) = ones(n,1);
else
    I(3:n+2,1) = w(:,2);
end

if(i == 3)
    inp_I = false(n+4,1);
    inp_I(3:n+3,1) = w(:,3);
else
    inp_I = [];
end

num_states = sum(I);
ind_states = find(I(:,1)==1);

X0 = zeros(n+3,1);
X0(I==1) = X;
X0(inp_I) = u;


% X = X0;

% n = n/2;
% 
% for i=1:n
%     x(i) = X(2*(i-1)+1); 
%     y(i) = X(2*(i-1)+2);
% end

% tic - timer

clearvars X


% N = n;
% X(1) = x(N);
% X(2:N+1,1) = x;
% X(N+2) = x(1);

% Y(2:N+1,1) = y;

dy = zeros(num_states,1);
for ind=3:num_states+2
    i = ind_states(ind-2);
    mu0 = mu(i-2);
    
    if(mod(i,2))
        dy(ind-2,1) = mu0*(X0(i)-X0(i)^3/3-X0(i+1)) ...
            + e*(X0(i+2) + X0(i-2) - 2*X0(i)); 
    else
        dy(ind-2,1) = X0(i-1)/mu0;
    end
end


