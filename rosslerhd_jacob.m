function dy = rosslerhd_jacob(X,e,w,a,b,c)
     N = length(X)/3;
     for i=1:N
         x(i) = X(3*(i-1)+1);
         y(i) = X(3*(i-1)+2);
         z(i) = X(3*(i-1)+3);
         
         wx(i) = w(3*(i-1)+1);
         wy(i) = w(3*(i-1)+2);
         wz(i) = w(3*(i-1)+3);
     end
     
     clearvars X
     X(1) = x(N);
     X(2:N+1) = x;
     X(N+2) = x(1);
     
     Y(2:N+1) = y;
     Z(2:N+1) = z;
     dy = zeros(N*3);
     for i=2:N+1
         dy(3*(i-1)+1,3*(i-1)+1) = -2*e;
         dy(3*(i-1)+1,3*(i)+1) = e;
         dy(3*(i-1)+1,3*(i-2)+1) = e;
         dy(3*(i-1)+1,3*(i-1)+2) = wy(i-1);
         dy(3*(i-1)+1,3*(i-1)+3) = -1;
         dy(3*(i-1)+2,3*(i-1)+1) = wy(i-1);
         dy(3*(i-1)+2,3*(i-1)+2) = a(i-1);
         dy(3*(i-1)+3,3*(i-1)+1) = Z(i);
         dy(3*(i-1)+3,3*(i-1)+3) = X(i)-c(i-1);
     end
     dy = dy(4:3*N+3,4:3*N+3);
end