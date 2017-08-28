function dy = rosslerhd(X,u,e,w,a,b,c)
i = size(w,2);
n = size(w,1);
I = false(n+4,1);
if(i==1)
    I(4:n+3,1) = ones(n,1);
else
    I(4:n+3,1) = w(:,2);
end

if(i == 3)
    inp_I = false(n+4,1);
    inp_I(4:n+3,1) = w(:,3);
else
    inp_I = [];
end

num_states = sum(I);
ind_states = find(I(:,1)==1);

X0 = zeros(n+4,1);
X0(I==1) = X;
X0(inp_I) = u;

clearvars X
dy = zeros(num_states,1);

for ind=4:num_states+3
    i = ind_states(ind-3);
    a0 = a(i-3);
    b0 = b(i-3);
    c0 = c(i-3);
    if(mod(i,3)==1)
         dy(ind-3,1) = -X0(i+1)-X0(i+2) + e*(X0(i+3)+X0(i-3)-2*X0(i));
    elseif(mod(i,3)==2)
        dy(ind-3,1) = X0(i-1) + a0*X0(i);
    else
         dy(ind-3,1) = b0 + X0(i)*(X0(i-2)-c0);
    end
end