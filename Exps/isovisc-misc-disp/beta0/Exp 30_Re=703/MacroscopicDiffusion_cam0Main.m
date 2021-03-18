close all
clear
clc
warning on
%im=image
%add=address
%dir=directory
add_dir = '/Users/aqib/Google Drive/DispFlow/Exps/isovisc-misc-disp/beta0/Exp 30_Re=703/'; % Dont forget to put \  also the Nx,...should be even

Gate_valve_image=37;
num_last_im_cam0= 215; 
NxStart=  100;
NxEnd=1680; 
NyStart=  7; 
NyEnd= 19;
Nxcenter= 18;  %Nxcenter is the location where the tape has been added to the downstream of the pipe xtape=1570 mm
Nxgate=1701; %the center of the gate valve indicated by a black line in the image xgate=0 mm
Framerate=8; 



Gate_valve_left = 1679; %leftmost pixel of gate valve
Gate_valve_right = 1725; % rightmost pixel of gate valve

lowerbound=0.1;
%===================================
make_nicer_edge_factor_X1=   1; 
make_nicer_edge_factor_X2=   1; 
num_first_im_cam0=  Gate_valve_image;
NumIntEvol=6;
V0=0;
Vf=0;
time_Vf=0;
Error_DiffusionOffset=0.0;
DiffusionOffset=0;
DiffusionCoeffEye=0;

%========================================================
% Delta_x_factor=1/0.9375; % This shows that each mm is for example 0.9325 pixel(de 22 a 34 : 0.9325)(de 35 a la fin : 0.9353)
% Pixelsize=642/701.675; %(pixel/mm)     625 pixels were in 701.675 mm
% Pixelsize=8.950012e-001;

%========================================================
Pixelsize=abs((Nxcenter-Nxgate)/(1570-0)); 
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

% Count=counter
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



figure(1)
colormap(gray)
Fig_ave_Final_im_cam0light=imagesc(ave_Final_im_cam0light);
set(gca,'XDir','reverse');
add_ave_cam0light=[add_dir,'Ave_cam0light.fig'];
% saveas(Fig_ave_Final_im_cam0light,add_ave_cam0light)


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

figure(2);
colormap(gray)
Fig_ave_Final_im_cam0black=imagesc(ave_Final_im_cam0black);
set(gca,'XDir','reverse');
add_ave_cam0black=[add_dir,'Ave_cam0black.fig'];
% saveas(Fig_ave_Final_im_cam0black,add_ave_cam0black)

count_cam0 = 0;  
for i_cam0=num_first_im_cam0:num_last_im_cam0
    count_cam0= count_cam0+1; 
    im_char_cam0= ['image (',int2str(i_cam0),')'];
    Final_add_im_cam0=[add_im_cam0,im_char_cam0];
    im_cam0(:,:,count_cam0) = double(imread(Final_add_im_cam0,'tif'));
    Final_im_cam0(:,:,count_cam0)=im_cam0(:,:,count_cam0)-0; 
end  

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
             if j > (Gate_valve_left - NxStart) && j < (Gate_valve_right - NxStart)
                 scaled_im_cam0(i,j,t)=0.5;
             end
        end
    end 
end

%Begin====Averaging the concentration accross the pipe 
% scaled_im_cam0=1-scaled_im_cam0;
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



figure(3)
axes('Position',[.2 .2 .7 .7])
box on
colormap(jet)
Fig_spatio_temporal=imagesc(xreal,treal,mean_scaled_im_cam0);
xlabel('$\hat{x}~(mm)$','Interpreter','latex','Fontsize',16);
ylabel('$\hat{t}~(s)$','Interpreter','latex','Fontsize',16);
colorbar
add_spatio_temporal_diffusion=[add_dir,'Spatio_temporal.fig'];




figure(4)
hold on
for t=1:timestep:count_cam0
    Fig_interface_evolution=plot(xreal,mean_scaled_im_cam0(t,:));
end

% axis([0 NxEnd 0 1])
box on
xlabel('$\hat{x}~(mm)$','Interpreter','latex','FontSize',16);
ylabel('$C$','Interpreter','latex','FontSize',16);
add_interface_evolution_diffusion=[add_dir,'Interface_evolution_Diffusuion.fig'];




%========== Find macroscopic concentraion diffusion equation===============


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

DiffusionCoeffFinal=mean(DiffusionCoeff)
Error_DiffusionCoeffFinal=std(DiffusionCoeff)%/DiffusionCoeffFinal;

DiffusionCoeffHorizontal=10^12
zeta_estimated=[-0.3:0.01:0.3];
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
figure(6)
colormap(jet)

     
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
    for i=2:length(mean_scaled_im_cam0(1,:))
        if mean_scaled_im_cam0(j,i)<lowerbound & checkpara==0
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
plot(time_Vf,Vf);
xlabel('$\hat{t}~(s)$','Interpreter','latex');
ylabel('$\hat{V}_f~(mm/s)$','Interpreter','latex');
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

timestep

mean_Vf_mm_per_sec=mean(Vf(round(length(Vf)/2):length(Vf)))
std_Vf_mm_per_sec=std(Vf(round(length(Vf)/2):length(Vf)))