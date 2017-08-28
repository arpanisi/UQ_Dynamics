function dy = lorenzhd_jacob(X,e,sigma,beta,rho)
     N = length(X)/3;
     for i=1:N
         x(i) = X(3*(i-1)+1);
         y(i) = X(3*(i-1)+2);
         z(i) = X(3*(i-1)+3);
     end
     
     clearvars X
     X(1) = x(N);
     X(2:N+1) = x;
     X(N+2) = x(1);
     
     Y(2:N+1) = y;
     Z(2:N+1) = z;
     dy = zeros(N+3);
     for i=2:N+1
         dy(3*(i-1)+1,3*(i-1)+1) = -2*e - sigma(i-1);
         dy(3*(i-1)+1,3*(i)+1) = e ;
         dy(3*(i-1)+1,3*(i-2)+1) = e;
         dy(3*(i-1)+1,3*(i-1)+2) = sigma(i-1);
         dy(3*(i-1)+2,3*(i-1)+1) = rho(i-1)-Z(i);
         dy(3*(i-1)+2,3*(i-1)+2) = -1;
         dy(3*(i-1)+2,3*(i-1)+3) = -X(i);
         dy(3*(i-1)+3,3*(i-1)+1) = Y(i);
         dy(3*(i-1)+3,3*(i-1)+2) = X(i);
         dy(3*(i-1)+3,3*(i-1)+3) = -beta(i-1);
     end
     dy = dy(4:3*N+3,4:3*N+3);
end