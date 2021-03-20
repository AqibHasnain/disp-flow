% This code takes three sets of images (a light reference, a dark
% reference, and the actual experiment) and applies Beer-Lambert Law to
% convert light intensity into concentration values. It is used to measure
% the velocities at the interface of the fluid flow and can be modified to
% check droplet sizes. It can also detect residual layers of fluid that are
% not necessarily visible in the unprocessed images. 

close all
clear
clc
warning on
%im=image
%add=address
%dir=directory
add_dir = '/Users/aqib/Desktop/github/disp-flow/Exps/isovisc-immisc-disp-unstable/beta60At125/Exp81GPH5/'; % directory where experiment images are stored

Gate_valve_image=67; % first image of experiment
num_last_im_cam0= 170; % last image of experiment
NxStart=  80;% x_pixel number you would like to start at, also Nx,...should be even
NxEnd=1783; % ending x_pixel number
NyStart=  6; 
NyEnd=  20;
Nxcenter= 83;  %Nxcenter is the location where the tape has been added to the downstream of the pipe xtape=1570 mm
Nxgate=1763; %the center of the gate valve indicated by a black line in the image xgate=0 mm
Framerate=12; % change this every time you change the framerate on camera

Gate_valve_left = 1738; %leftmost pixel of gate valve
Gate_valve_right = 1784; % rightmost pixel of gate valve

lowerbound=0.90; % boundary used to track interface of fluids. It is based on concentration values
%===================================
make_nicer_edge_factor_X1=   1; 
make_nicer_edge_factor_X2=   1; 
num_first_im_cam0=  Gate_valve_image;
NumIntEvol=8; % how many snapshots would you like in Fig 6.
V0=81.4; %mm/s
Vf=0;
time_Vf=0;
Error_DiffusionOffset=0.0; %use this for diffusive flows if you would like to fit an error function
DiffusionOffset=0;%see above
DiffusionCoeffEye=0;%see above

%========================================================
% Delta_x_factor=1/0.9375; % This shows that each mm is for example 0.9325 pixel(de 22 a 34 : 0.9325)(de 35 a la fin : 0.9353)
% Pixelsize=642/701.675; %(pixel/mm)     625 pixels were in 701.675 mm
% Pixelsize=8.950012e-001;

%========================================================
Pixelsize=abs((Nxcenter-Nxgate)/(1570-0)); %converting pixels to mm
timestep=round((num_last_im_cam0-num_first_im_cam0)/NumIntEvol);
make_nicer_edge_factor_X1=1; % on the left it drops a number of pixels (dif de 0)
make_nicer_edge_factor_X2=1; % It drops a number of pixels on the right(dif de 0)
%========================================================


Delta_x_factor=1/Pixelsize;
Delta_t_factor=1/Framerate;

add_im_cam0=[add_dir,'cam0 1400/']; 
add_im_cam0light=[add_dir,'cam0light/'];
add_im_cam0black=[add_dir,'cam0black/'];


filelist_cam0 = dir(add_im_cam0);
filelist_cam0light = dir(add_im_cam0light); 
filelist_cam0black = dir(add_im_cam0black); 

% loading light reference images
count_cam0light = 1; % Initial value for the counter  
ave_Final_im_cam0light=0; % Initial value 
for i_cam0light=1:size(filelist_cam0light,1)-3         
    im_char_cam0light= ['image (',int2str(i_cam0light),')']; % makes a character named "image"
    Final_add_im_cam0light=[add_im_cam0light,im_char_cam0light];
    im_cam0light(:,:,count_cam0light) = double(imread(Final_add_im_cam0light,'tif'));
    Final_im_cam0light(:,:,count_cam0light)=im_cam0light(:,:,count_cam0light)-0;
    ave_Final_im_cam0light=Final_im_cam0light(:,:,count_cam0light)+ave_Final_im_cam0light;
    count_cam0light= count_cam0light+1; 
end
ave_Final_im_cam0light=ave_Final_im_cam0light/(count_cam0light-1);


% light reference image
figure(1)
colormap(gray)
Fig_ave_Final_im_cam0light=imagesc(ave_Final_im_cam0light);
set(gca,'XDir','reverse');
add_ave_cam0light=[add_dir,'Ave_cam0light.fig'];
% saveas(Fig_ave_Final_im_cam0light,add_ave_cam0light)

% loading black reference image
count_cam0black = 1;  
ave_Final_im_cam0black=0;
for i_cam0black=1:size(filelist_cam0black,1)-3         
    im_char_cam0black= ['image (',int2str(i_cam0black),')'];
    Final_add_im_cam0black=[add_im_cam0black,im_char_cam0black];
    im_cam0black(:,:,count_cam0black) = double(imread(Final_add_im_cam0black,'tif'));
    Final_im_cam0black(:,:,count_cam0black)=im_cam0black(:,:,count_cam0black)-0;
    ave_Final_im_cam0black=Final_im_cam0black(:,:,count_cam0black)+ave_Final_im_cam0black;
    count_cam0black= count_cam0black+1; 
end
ave_Final_im_cam0black=ave_Final_im_cam0black/(count_cam0black-1);

% black reference image
figure(2);
colormap(gray)
Fig_ave_Final_im_cam0black=imagesc(ave_Final_im_cam0black);
set(gca,'XDir','reverse');
add_ave_cam0black=[add_dir,'Ave_cam0black.fig'];
% saveas(Fig_ave_Final_im_cam0black,add_ave_cam0black)

% loading experiment images
count_cam0 = 0;  
for i_cam0=num_first_im_cam0:num_last_im_cam0
    count_cam0= count_cam0+1; 
    im_char_cam0= ['image (',int2str(i_cam0),')'];
    Final_add_im_cam0=[add_im_cam0,im_char_cam0];
    im_cam0(:,:,count_cam0) = double(imread(Final_add_im_cam0,'tif'));
    Final_im_cam0(:,:,count_cam0)=im_cam0(:,:,count_cam0)-0; 
end  

% cropping images 
Cropped_im_cam0=double(Final_im_cam0(NyStart:NyEnd,NxStart:NxEnd,:));
Cropped_ave_im_cam0light=double(ave_Final_im_cam0light(NyStart:NyEnd,NxStart:NxEnd));
Cropped_ave_im_cam0black=double(ave_Final_im_cam0black(NyStart:NyEnd,NxStart:NxEnd));


% Beer lambert law
% C={log(I(C))-log(I(Cmin))}/{log(I(Cmax))-log(I(Cmin))}
for t=1:count_cam0
    scaled_im_cam0(:,:,t)=abs(log(Cropped_im_cam0(:,:,t))-log(Cropped_ave_im_cam0black))./(log(Cropped_ave_im_cam0light)-log(Cropped_ave_im_cam0black));
    for i=1:size(scaled_im_cam0(:,:,t),1)
        for j=1:size(scaled_im_cam0(:,:,t),2)
            if scaled_im_cam0(i,j,t)<0
                scaled_im_cam0(i,j,t)=0;
            end
            if scaled_im_cam0(i,j,t)>1
                scaled_im_cam0(i,j,t)=1;
            end
             if j > (Gate_valve_left - NxStart) && j < (Gate_valve_right - NxStart) % covering gate valve so it looks better
                 scaled_im_cam0(i,j,t)=0.5;
             end
        end
    end 
end

%Begin====Averaging the concentration accross the pipe 
scaled_im_cam0=1-scaled_im_cam0;
mean_scaled_im_cam0=0;
treal(1)=(num_first_im_cam0-Gate_valve_image)*1/Framerate;
xreal(1)=0-(NxEnd-Nxgate)*1/Pixelsize;   
for i=NxEnd-NxStart:-1:1
    for t=count_cam0:-1:1
        treal(t)=treal(1)+(t-1)*1/Framerate;
        xreal((NxEnd-NxStart)-i+1)=xreal(1)+((NxEnd-NxStart)-i)*1/Pixelsize;
        mean_scaled_im_cam0(t,(NxEnd-NxStart)-i+1)=mean(scaled_im_cam0(:,i,t));
    end
end
%End======Averaging the concentration accross the pipe


% Spatiotemporal Diagram 
figure(3)
axes('Position',[.2 .2 .7 .7])
box on
colormap(jet)
Fig_spatio_temporal=imagesc(xreal,treal,mean_scaled_im_cam0);
xlabel('$\hat{x}~(mm)$','Interpreter','latex','Fontsize',16);
ylabel('$\hat{t}~(s)$','Interpreter','latex','Fontsize',16);
title('$\beta=60^{\circ}$','Interpreter','latex');
%set(gca,'XTick',[0 350 700 1050 1400])
%set(gca,'YTick',[0 4 8 12])
set(gca,'FontSize',12)
set(gca,'linewidth',2)
box on
c = colorbar
c.LineWidth = 1.5;
c.Ticks = [0,0.2,0.4,0.6,0.8,1];
c.FontSize = 12;
add_spatio_temporal_diffusion=[add_dir,'Spatio_temporal.fig'];








% C vs.x
figure(4)
axes('Position',[.2 .2 .7 .7])
hold on
for t=1:timestep:count_cam0
    Fig_interface_evolution=plot(xreal,mean_scaled_im_cam0(t,:),'Color',[0,0,0.8]);
end
xred1 = [0:1600];
yred1 = ones(size(xred1,2))*0.05; plot(xred1,yred1,'r','linewidth',2);


% xg1 = [20:100];
% yg1 = ones(size(xg1,2))*0.9; plot(xg1,yg1,'k','linewidth',2);
% xg2 = [120:200];
% yg2 = ones(size(xg2,2))*0.9; plot(xg2,yg2,'k','linewidth',2);
% xg3 = [220:300];
% yg3 = ones(size(xg3,2))*0.9; plot(xg3,yg3,'k','linewidth',2);
% xg4 = [320:400];
% yg4 = ones(size(xg4,2))*0.9; plot(xg4,yg4,'k','linewidth',2);
% xg5 = [420:500];
% yg5 = ones(size(xg5,2))*0.9; plot(xg5,yg5,'k','linewidth',2);
% xg6 = [520:600];
% yg6 = ones(size(xg6,2))*0.9; plot(xg6,yg6,'k','linewidth',2);
% xg7 = [620:700];
% yg7 = ones(size(xg7,2))*0.9; plot(xg7,yg7,'k','linewidth',2);
% xg8 = [720:800];
% yg8 = ones(size(xg8,2))*0.9; plot(xg8,yg8,'k','linewidth',2);
% xg9 = [820:900];
% yg9 = ones(size(xg9,2))*0.9; plot(xg9,yg9,'k','linewidth',2);
% xg10 = [920:1000];
% yg10 = ones(size(xg10,2))*0.9; plot(xg10,yg10,'k','linewidth',2);
% xg11 = [1020:1100];
% yg11 = ones(size(xg11,2))*0.9; plot(xg11,yg11,'k','linewidth',2);
% xg12 = [1120:1200];
% yg12 = ones(size(xg12,2))*0.9; plot(xg12,yg12,'k','linewidth',2);
% xg13 = [1220:1300];
% yg13 = ones(size(xg13,2))*0.9; plot(xg13,yg13,'k','linewidth',2);
% xg14 = [1320:1400];
% yg14 = ones(size(xg14,2))*0.9; plot(xg14,yg14,'k','linewidth',2);
% xg15 = [1420:1500];
% yg15 = ones(size(xg15,2))*0.9; plot(xg15,yg15,'k','linewidth',2);
% xg16 = [1520:1600];
% yg16 = ones(size(xg16,2))*0.9; plot(xg16,yg16,'k','linewidth',2);
% xlim([0 1600])
% set(gca,'XTick',[0 400 800 1200 1600])
% set(gca,'YTick',[0 0.2 0.4 0.6 0.8 1.0])
box on
set(gca,'FontSize',12)
set(gca,'linewidth',2)
xlabel('$\hat{x}~(mm)$','Interpreter','latex','FontSize',16);
ylabel('$C$','Interpreter','latex','FontSize',16);
add_interface_evolution_diffusion=[add_dir,'Interface_evolution_Diffusuion.fig'];
hold off



%========== Find macroscopic concentraion diffusion equation===============

% C vs. (x-V0t)/(sqrt(t)-offset)
figure(5)
hold on
%  x=[length(mean_scaled_im_cam0(1,:)):-1:1];
 counter=0;
for t=count_cam0:-1:1   
    counter=counter+1;
    Fig_interface_evolution_Diffusion=plot(1e-3*(xreal-V0*treal(t))./sqrt(treal(t))-DiffusionOffset,mean_scaled_im_cam0(t,:));
%         Fig_interface_evolution_Diffusion=plot(1e-3*(xreal)./(treal(t))-DiffusionOffset,mean_scaled_im_cam0(t,:));

    DDD=mean(((1e-3*(xreal-V0*treal(t))./sqrt(treal(t))-DiffusionOffset)./(2*erfinv(1-2*mean_scaled_im_cam0(t,:)))).^2);
    DiffusionCoeff(counter)=DDD;
end

DiffusionCoeffFinal=mean(DiffusionCoeff);
Error_DiffusionCoeffFinal=std(DiffusionCoeff);%/DiffusionCoeffFinal;

DiffusionCoeffHorizontal=10^12;
zeta_estimated=[-0.1:0.01:0.1];
C_estimated=1/2*(1-erf(1./(2*sqrt(DiffusionCoeffHorizontal))*(zeta_estimated)));
plot(zeta_estimated,C_estimated,'r');

C_estimated=1/2*(1-erf(1./(2*sqrt(DiffusionCoeffEye))*(zeta_estimated)));
plot(zeta_estimated,C_estimated,'r');


box on
xlim([-0.3 0.3]);
% xlabel('$(\hat{x}-\hat{V_0}\hat{t})/\sqrt{\hat{t}}$','Interpreter','latex','FontSize',16);
xlabel('$(\hat{x}-\hat{V0t})/\sqrt{t}$','Interpreter','latex','FontSize',16);

ylabel('$C$','Interpreter','latex','FontSize',16);
add_interface_evolution_diffusion_scaled=[add_dir,'Interface_evolution_Diffusuion_Scaled.fig'];


saveas(Fig_spatio_temporal,add_spatio_temporal_diffusion)
saveas(Fig_interface_evolution,add_interface_evolution_diffusion)
saveas(Fig_interface_evolution_Diffusion,add_interface_evolution_diffusion_scaled)

% =========================================================================
% =========================================================================
% color snapshots
figure(6)
colormap(jet) % gray used for miscible flows, jet for immiscible

     
i=0;
for t=1:timestep:num_last_im_cam0-num_first_im_cam0+1
    i=i+1;
    h=subplot(NumIntEvol+1,1,i);
    p=get(h,'pos');
    p(2)=p(2)+0.05;
    set(h,'pos',p);
    imagesc(scaled_im_cam0(:,:,t));

    
%     imcontrast
    caxis([0 1]);
    set(gca,'XDir','reverse');
    set(gca,'xtick',[]);set(gca,'ytick',[]);
end

% annotation('textbox',[0.0492142857142857 0.0285714285714286 0.0900714285714286 0.0714285714285715],'Interpreter','latex',...
%     'String',{'a)'},'FontSize',16,'FitBoxToText','off','EdgeColor','none');

h=subplot(NumIntEvol+1,1,NumIntEvol+1);
hold off
p=get(h,'pos');
p(2)=p(2)+0.05;
set(h,'pos',p);
    
Conc_Colorbar=zeros(NyEnd-NyStart,1000);
for i=1000:-1:1
    Conc_Colorbar(:,1000-i+1)=(i-1)/999;
end
   Fig_DispType=imagesc(Conc_Colorbar);
   caxis([0 1])
   set(gca,'XDir','reverse');
   set(gca,'ytick',[]);
   set(gca,'xtick',[0.5 100 200 300 400 500 600 700 800 900 1000.5]);
   set(gca,'XTickLabel',[1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0])

add_cam0_DispType=[add_dir,'cam0_DispType.fig'];
saveas(Fig_DispType,add_cam0_DispType)

% =========================================================================
%=======================================================

% Find the stretch length
checkpara=0;
dtPara=1;
for j=1:length(mean_scaled_im_cam0(:,1)) 
    checkpara=0; % It's only a check parameter
    for i=length(mean_scaled_im_cam0(1,:)):-1:2 %2:length(mean_scaled_im_cam0(1,:))
        if mean_scaled_im_cam0(j,i)>lowerbound & checkpara==0
            x_contact1(j)=xreal(i); % Find the contact point for the lower limit
            checkpara=1;
            x_contact1_real(j)=x_contact1(j);
            
%             Slope1=(mean_scaled_im_cam0(j,i)-mean_scaled_im_cam0(j,i-1))/(xreal(i)-xreal(i-1));
%             Yintercept1=-Slope1*xreal(i-1)+mean_scaled_im_cam0(j,i-1);
%             x_contact1_real(j)=(lowerbound-Yintercept1)/Slope1;
            break
        end        
    end

    
    if j>dtPara   & checkpara==1     
%             Vf(j)=(x_contact1_real(j)-x_contact1_real(j-dtPara))/(dtPara*0.25);%/V0*0.01905; %Vf=dx/dt and real dt=0.5 s; width=0.01905 m
            Vf(j)=(x_contact1_real(j)-x_contact1_real(1))/((j-1)*dtPara*(1/Framerate));%/V0*0.01905; %Vf=dx/dt and real dt=0.5 s; width=0.01905 m
            time_Vf(j)=treal(j);
    end

end

 % See

Fig_Vf_vs_time=figure(7);
axes('Position',[.2 .2 .7 .7])
box on
plot(time_Vf,Vf,'Color',[0,0,0.75],'LineWidth',1.25);
set(gca,'FontSize',12)
set(gca,'linewidth',2)
box on
xlabel('$\hat{t}~(s)$','Interpreter','latex','FontSize',16);
ylabel('$\hat{V}_f~(mm/s)$','Interpreter','latex','FontSize',16);
add_Fig_Vf_vs_time=[add_dir,'Fig_Vf_vs_time_cam0.fig'];
saveas(Fig_Vf_vs_time,add_Fig_Vf_vs_time);


%=======================================================

% =========================================================================

add_Report=[add_dir,'Report_cam0_Diffusion.txt'];
fileReport=fopen(add_Report,'w');
fprintf('\n \n');
fprintf(fileReport,'Gate_valve_image=%4.0f \n',Gate_valve_image);
fprintf(fileReport,'Gate_valve_left=%4.0f \n',Gate_valve_left);
fprintf(fileReport,'Gate_valve_right=%4.0f \n',Gate_valve_right);
fprintf(fileReport,'num_first_im_cam0=%4.0f \n',num_first_im_cam0);
fprintf(fileReport,'num_last_im_cam0=%4.0f \n',num_last_im_cam0);
fprintf(fileReport,'NxEnd=%4.0f \n',NxEnd);
fprintf(fileReport,'Nxcenter=%4.0f \n',Nxcenter);
fprintf(fileReport,'Nxgate=%4.0f \n',Nxgate);
fprintf(fileReport,'NxStart=%4.0f \n',NxStart);
fprintf(fileReport,'NyStart=%4.0f \n',NyStart);
fprintf(fileReport,'NyEnd=%4.0f \n',NyEnd);
fprintf(fileReport,'Framerate=%4.0f \n',Framerate);
fprintf(fileReport,'NumIntEvol=%4.0f \n',NumIntEvol);
fprintf(fileReport,'Pixelsize=%e \n',Pixelsize);
fprintf(fileReport,'timestep=%4.0f \n',timestep);
fprintf(fileReport,'lowerbound=%e \n',lowerbound);
fprintf(fileReport,'make_nicer_edge_factor_X1=%4.0f \n',make_nicer_edge_factor_X1);
fprintf(fileReport,'make_nicer_edge_factor_X2=%4.0f \n',make_nicer_edge_factor_X2);
fprintf(' \n');
fprintf(' \n');
fprintf(fileReport,'V0=%f \n',V0);  
fprintf(fileReport,'DiffusionOffset=%f \n',DiffusionOffset); 
fprintf(fileReport,'Error_DiffusionOffset=%f \n',Error_DiffusionOffset); 
fprintf(fileReport,'DiffusionCoeffEye=%e \n',DiffusionCoeffEye);
% fprintf(fileReport,'Vf=%f \n',Vf);       
% fprintf(fileReport,'err_in_Vf=%f \n',err_in_Vf);
fprintf('\n \n');
fclose(fileReport);

timestep;

mean_Vf_mm_per_sec=mean(Vf(round(length(Vf)/2):length(Vf)))
std_Vf_mm_per_sec=std(Vf(round(length(Vf)/2):length(Vf)))

% scaled_im_cam0 contains the dynamics that we want for DMD
save('scaled_images.mat','scaled_im_cam0');