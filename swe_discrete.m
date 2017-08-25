%% Shallow Water Model with constant L and MaxT
% Changing dx and dt
clear;clc;
foldername_2  = strcat('Results/swe');


N_theta = 10;

Ind_N = 80;



L = 60e3;

[Tstep,DN] = meshgrid(150:3:1200,60:3:1200);

% X_1_length_mean = zeros(size(DN));
% X_1_length_cov = zeros(size(DN));

% filename = strcat(foldername,'/L',num2str(L/1000),'.mat');


cf = 0.0002; % friction constant in 1/sec
g = 9.81; % acceleration due to gravity
MaxT = 48e3;



DX = L./(DN+0.5);
DT = MaxT./Tstep;

count = 0;

% Err_mesh_mean = cell(3,1);
% Err_mesh_cov = cell(3,1);
% for i=1:3
%     Err_mesh_mean{i,1} = zeros(size(DT));
%     Err_mesh_cov{i,1} = zeros(size(DT));
% end

for mesh_i=1%:size(DN,1)
    mesh_i
    for mesh_j = 1%:size(DN,2)
        mesh_j
        
        
        
        
        dt = DT(mesh_i,mesh_j);
        dx = DX(mesh_i,mesh_j);
        N = DN(mesh_i,mesh_j);
        dim = 2*(N+1);
        dx_2 = L/N;
        MaxTime = round(MaxT/dt);
        
        U_cl_data = cell(length(Theta),3);
        U_data = cell(length(Theta),1);
        
        
        row1 = [-g/(2*dx) 1/dt+cf/2 g/(2*dx)];
        
        row3 = [g/(2*dx) 1/dt-cf/2 -g/(2*dx)];
        
        temp_x_1 = zeros(size(Theta));
        
        U_mean_cl = cell(3,1);
        U_cov_cl = cell(MaxTime+1,3);
        
%         foldername_2 = strcat(foldername,'/dim',num2str(N));
        
%         for theta_i = 1:length(Theta)
%             theta_i
            theta = 0;% Theta(theta_i);%40/(60e3);
            % theta = Theta(theta_i);
            

            
            %% Clustering algorithm for each realization
            
            
            
            if(~exist(foldername_2,'dir'))
                mkdir(foldername_2)
            end
            
            D_bar = eye(2*(N+1));
            
            A_bar = zeros(2*(N+1));
            
            for ind_row = 1:N
                theta = 0;%Theta(ind_row);
                D = mu(ind_row);
                row2 = [-D/(2*dx) 1/dt D/(2*dx)+theta/2];
                row4 = [D/(2*dx) 1/dt -D/(2*dx)-theta/2];
                r_1 = ind_row*2;
                r_2 = ind_row*2+1;
                c_1 = ind_row*2-1;
                c_2 = ind_row*2;
                D_bar(r_1,c_1:c_1+2) = row1;
                D_bar(r_2,c_2:c_2+2) = row2;
                
                A_bar(r_1,c_1:c_1+2) = row3;
                A_bar(r_2,c_2:c_2+2) = row4;
            end
            B = zeros(2*(N+1),1);
            B(1,1) = 1;
            
            
            y_cl = linspace(0,L,N + 1);
            u_0 = 0 + sin (pi * y_cl(1:N+1)/L );
            x0 = zeros(2*(N+1),1);

            
            x02 = x0;
            
            A_sl = pinv(D_bar)*A_bar;
            % A_sl = A_sl(2:end-1,2:end-1);
            P = abs(A_sl);
            P(P<1e-3) = 0;
            P(1,1) = 1;
            P(end,end) = 1;
            
            com = fast_mo(P);
            %     com = reshape(kron((1:dim/10),ones(10,1)),dim-2,1);
            com(dim-1:dim,1) = com(dim-2);
            num_cl = length(unique(com));
            Cl = zeros(dim,num_cl);
            D_Cl = sum(P,2);
            
            
            for i=1:num_cl
                dom_cl = com==i;
                for j=1:dim
                    Cl(j,i) = sum(P(j,dom_cl))/D_Cl(j);
                end
            end
            
            Cl(Cl<1e-3) = 0;
            Cl( :, ~any(Cl,1) ) = [];
            Cl = Cl./(sum(Cl,2)*ones(1,num_cl));
            
            %     Cl = commDetNMF(P);
            
            x_1_length = length(find(com == 1))*L/(dim*1000);
            
            temp_x_1(theta_i) = x_1_length;
            
            %  store(x_1_length,X_1_length,mesh_i,mesh_j)
            
            
%             Cl_x = Cl(1:2:end,:);
%             y_cl = linspace(0,L,N + 1)/1000;
%             x_cl = 1:num_cl;
            
%             figure
%             imagesc(x_cl,y_cl,Cl_x)
%             colorbar
%             xlabel('Cluster Number')
%             ylabel('Distance (km)')
             title(strcat('Cluster Strcture for N = ',num2str(N),...
                 ' dx =',num2str(dx),' and dt = ',num2str(dt)),'FontSize',20)
%             filename = strcat(foldername_2,'/fig_cluster_',...
%                 num2str(dim),'_',num2str(L/1000),'_',num2str(dt));
%             print(gcf,'-dpng',filename)
            
%             MaxTime = round(MaxT/dt);
            t_low = floor(MaxTime/16);
            u = [1/(MaxTime-1)*((1:t_low)-1) 1/(MaxTime-1)*(MaxTime/16-1)*ones(1,MaxTime-t_low+1)];
            C = zeros(dim,1);
            C(1:2:end) = 1;
            C = diag(C);
            C = C(1:2:end,:);
            sys = ss(pinv(D_bar)*A_bar,pinv(D_bar)*B,C,[],dt);
            U = lsim(sys,u,0:dt:MaxT,x0);
            U(U<0)=0;
            
                        
            for type = 1:3
                                
                if (type == 3)
                    Cl = zeros(dim,num_cl);
                    for i=1:dim
                        idx = com(i);
                        Cl(i,idx) = 1;
                    end
                end
                
                if (type == 2)
                    Cl_2 = zeros(dim,num_cl);
                    for i=1:dim
                        idx = com(i);
                        Cl_2(i,idx) = 1;
                    end
                end
                
                x0 = x02;
                X_cl = zeros(2*(N+1),MaxTime);
                U_cl = zeros(N+1,MaxTime+1);
                U_cl(:,1) = x0(1:2:end,1);
                %         U_cl(:,1) = U(1,:)';
                for indt = 0:MaxTime
                    
                    
                    uk = 1/(MaxTime-1)*(MaxTime/16-1);
                    
                    
                    if indt < MaxTime/16
                        
                        uk = 1/(MaxTime-1)*(indt-1);
                        
                    end
                    X_k_cl = zeros(dim,num_cl);
                    for cl_i = 1:num_cl
                        if(type == 3 || type == 1)
                            cl = logical(Cl(:,cl_i));
                            A_bar_cl = A_bar(cl,cl);
                            D_bar_cl = D_bar(cl,cl);
                            B_cl = B(cl);
                            x0_cl = x0(cl);
                            x_k_cl = pinv(D_bar_cl)*A_bar_cl*x0_cl + pinv(D_bar_cl)*B_cl*uk;
                            X_k_cl(cl,cl_i) = x_k_cl;
                        else
                            cl = Cl(:,cl_i);
                            cl1 = find(cl>0.1);
                            cl2 = find(cl<=0.1 & cl > 0);
                            A_bar_cl = A_bar(cl1,cl1);
                            D_bar_cl = D_bar(cl1,cl1);
                            B_cl = B(cl1);
                            x0_cl = x0(cl1);
                            u_cl = x0(cl2);
                            B_u_cl = A_bar(cl1,cl2);
                            x_k_cl = pinv(D_bar_cl)*A_bar_cl*x0_cl + ...
                                pinv(D_bar_cl)*B_cl*uk + pinv(D_bar_cl)*B_u_cl*u_cl;
                            X_k_cl(logical(cl),cl_i) = [x_k_cl; u_cl];
                        end
                    end
                    if(type == 1 || type == 3)
                        x_k = sum((X_k_cl.*Cl),2);
                    else
                        x_k = sum((X_k_cl.*Cl_2),2);
                    end
                    % x_k(1,1) = 0;
                    x_k(end,1) = 0;
                    
                    U_cl(:,indt+1) = x_k(1:2:end);
                    
                    X_cl(:,indt+1) = x_k;
                    x0 = x_k;
                end
                
                U_cl = U_cl';
                
                U_cl(U_cl<0)=0;
                
                % U_cl_cell{theta_i,1} = U_cl;
                U_cl_data{theta_i,type} = U_cl;
            end
            close all
            
            U_data{theta_i,1} = U;
        end
        
        [mean_x_1,cov_x_1] = UT(temp_x_1,W_Theta);
        
        X_1_length_mean(mesh_i,mesh_j) = mean_x_1; %Storing the average x_1 length
        X_1_length_cov(mesh_i,mesh_j) = cov_x_1; %Storing the covariance of x_1 length
        
        [U_mean,P_U_mean] = UT_cell(U_data,W_Theta);
        P_U_mean = P_U_mean';
        
        for type = 1:3
            
            [temp_mean_cl,temp_P_cl] = UT_cell(U_cl_data(:,type),W_Theta);
            temp_P_cl = temp_P_cl';
            U_mean_cl{type,1} = temp_mean_cl;
            U_cov_cl(:,type) = temp_P_cl';
            
            err_mean = zeros(MaxTime+1,1);
            err_cov = zeros(MaxTime+1,1);
            
            parfor ti = 1:MaxTime+1
                err_mean(ti) = norm(U_mean(ti,:)-temp_mean_cl(ti,:))...
                    /(norm(U_mean(ti,:)+eps)*(N+1));
                err_cov(ti) = norm(P_U_mean{1,ti}-temp_P_cl{1,ti})...
                    /(norm(P_U_mean{1,ti}+eps)*(N+1));
            end
            
%             err_cl = abs(U_mean - U_mean_cl{type,1});
%             err_cl_mesh = sum(sum(err_cl,2))/((N+1)*(MaxT+1));
            Err_mesh_mean{type,1}(mesh_i,mesh_j) = mean(err_mean);
            Err_mesh_cov{type,1}(mesh_i,mesh_j) = mean(err_cov);
            
%             figure
            
%             [X_plot,Y_plot] = meshgrid(0:dx:L,0:dt:dt*MaxT);
%             subplot(1,2,1)
%             surf(X_plot,Y_plot,U_mean)
%             xlabel('Length')
%             ylabel('time')
%             title('True')
%             subplot(1,2,2)
% %             surf(X_plot,Y_plot,U_mean_cl{type,1})
%             surf(X_plot,Y_plot,err_cl)
%             title('Estimate')
%             xlabel('Length')
%             ylabel('time')
%             filename = strcat(foldername_2,'/fig_comp_',num2str(dim),...
%                 '_',num2str(type));
            %             print(gcf,'-dpng',filename)
        end
        
%         close all
        
        %         if(mesh_j > 1)
        %             conv = (U_cell_interp{mesh_j-1,1}-U_cell_interp{mesh_j,1})./U_cell_interp{mesh_j-1,1};
        %             conv(isnan(conv))=0;
        %             Conv(mesh_j-1) = norm(conv);
    end
    filename = 'temp_3.mat';
    save(filename,'Err_mesh_mean','Err_mesh_cov',...
        'X_1_length_mean','X_1_length_cov','Tstep','DN');
end

h = figure;
surf(Tstep,DN,X_1_length_cov)
xlabel('Temporal Discretization')
ylabel('Spatial Discretization')
title('Mean length in Clustering')
filename = strcat(foldername_2,'/fig_conv_clust_length');
print(h,'-dpng',filename)

h = figure;
surf(Tstep,DN,X_1_length_mean)
xlabel('Temporal Discretization')
ylabel('Spatial Discretization')
title('Mean length in Clustering')
filename = strcat(foldername_2,'/fig_mean_clust_length');
print(h,'-dpng',filename)

h = figure;
for i=1:3
    subplot(1,3,i)
    surf(Tstep,DN,Err_mesh_mean{i,1})
    xlabel('Temporal Discretization')
    ylabel('Spatial Discretization')
    title(strcat('Error in Mean for Clustering ',num2str(i)))
end
filename = strcat(foldername_2,'/fig_mean_error');
print(h,'-dpng',filename)

h = figure;
for i=1:3
    subplot(1,3,i)
    surf(Tstep,DN,Err_mesh_cov{i,1})
    xlabel('Temporal Discretization')
    ylabel('Spatial Discretization')
    title(strcat('Error in Convergence for Clustering ',num2str(i)))
end
filename = strcat(foldername_2,'/fig_conv_error');
print(h,'-dpng',filename)
% figure
% % DT_f = dt_f*300;
% scatter(DX,X_1_length,'filled')
% hold on
% myfit = polyfit(DX,X_1_length',1);
% x_est = myfit(2) + myfit(1)*DX;
% plot(DX,x_est,'Linewidth',2,'Color','black')
% xlabel(num2str(dt))
% ylabel('Cluster Length')
% title('dt vs cluster length')
% filename = strcat(foldername_2,'/fig_clusterdx_',...
%         num2str(dim));
% print(gcf,'-dpng',filename)

% end

% end

% save('temp3','U_cell','U_cell_interp','Conv')

% plot(DN(2:end),Conv,'Linewidth',2)
% xlabel('Discrete Points')
% ylabel('Convergence')
% title('Convergence vs Resolution')
% print(gcf,'-dpng','fig3')

% close all