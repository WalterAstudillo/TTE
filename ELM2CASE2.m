close all; clear all; clc; 
%% elegir el case para ejecutar  2
p=2;  % 1=potencia , 2=voltaje 3=lambda
caserun = case300; 
%caserun = case500; 
caserun = case3375wp;
vargauss=0.05; % varianza en porcentaje
% ELM2CASE
%% branch data
%  r, resistance (p.u.)  x, reactance (p.u.) b, total line charging susceptance (p.u.)
%  Z=r+jx
%  Y =1/Z
%  b, la parte imaginaria de la admitnacia Y
%  v = v / norm(v); normalizar valores 0-1
%% ejecutamos el runopf MATPOWER 
copf = runopf(caserun); % ejecutamos el caso
base=caserun.baseMVA; % potencia base 
%% Datos de entrada [MATRIZ DE DATOS]
opfPg = (caserun.bus(:,3))/(base); % tomamos la columba de Pd p.u 
opfQg = (caserun.bus(:,4))/(base); % tomamos la columba de Qd p.u
magnitudV = copf.bus(:,8); % tomamos la magnitud del voltaje p.u
angulo = (copf.bus(:,9))*(pi/180); % tomamos el angulo deg
LambdaP = copf.bus(:,14);%Lambda($/MVA-hr) Potencia activa
LambdaQ = copf.bus(:,15);%Lambda($/MVA-hr) Potencia reactiva
r = caserun.branch(:,3);
x = (ones(length(caserun.branch(:,4)),1)*1j).*caserun.branch(:,4);%x=caserun.branch(:,4);
b = caserun.branch(:,5);
Zpu = r+x;  % impedancia p.u
Ypu = 1./Zpu; % matriz de admitancia de bus Y 
%% Normalizar datos
opfPg = opfPg/norm(opfPg);
opfQg = opfQg/norm(opfQg);
magnitudV = magnitudV/norm(magnitudV);
angulo = angulo/norm(angulo);
LambdaP = LambdaP/norm(LambdaP);
LambdaQ = LambdaQ/norm(LambdaQ);
r = r/norm(r);
x = x/norm(x);
b = b/norm(b);
Zpu = Zpu/norm(Zpu);
Ypu = Ypu/norm(Ypu);
%% Conjunto de datos normalizados
if p==1
xinput = [opfPg,opfQg]; % matriz de potencia input
elseif p==2
xinput = [magnitudV,angulo]; % matriz de magnitud y angulo output OPF
elseif p==3
xinput = [LambdaP, LambdaQ];     
end
%% NNeuronal 
% Son el número de unidades o neuronas para generar el mapeo en la capa 
% oculta de 'SLFN' (n_hidden = [256, 400, 512, 1024, 2048] para la red de
% 500 buses y [4096, 5116, 6144,7168] para la red de bus 4918)
%NNeuronal=(length(opfPg))*2;  % validamos el numero de neuronas a la entrada
%% first 80% of the data is used for training 
Ns = floor(0.8*length(xinput));        
% training data
xtrain = xinput(1:Ns,:); % input              
% testing data
xtest = xinput(Ns+1:end,:);  %20% de prueba del sistema          
% z-score normalization
Neurons_HL=35; % 35 varia de numero 
%Neurons_HL = 6*size(xtrain,2);
%Neurons_HL =(length(opfPg))*2;  % otra opccion

%% inicio de entrenamiento 
[xtrain,mux,sigmax] = zscore(xtrain);       
[xtest,muy,sigmay] = zscore(xtest);
% Neurons in the hidden layer
start_time_train=cputime; % tiempo de entrenamiento 
Input_Features=size(xtrain,2);
Inputweights=rand(Neurons_HL,Input_Features)*2-1; % randomly generated input weights
% randomly generated biases
Bias_HL=rand(Neurons_HL,1);
Biasmatrix=Bias_HL(:,ones(1,Ns));    
% output of the hidden layer
Prod=xtrain*Inputweights';
H=Prod+Biasmatrix'; 
%% chossing activation function
AF='sig';  % por la potencia Q en ocaciones es negativa           
if strcmp(AF,'tanh')
    Hout=tanh(H);
elseif strcmp(AF,'sig')
    Hout=1./(1+exp(-H));
elseif strcmp(AF,'sin')
    Hout=sin(H);
elseif strcmp(AF,'cos')
    Hout=cos(H);
elseif strcmp(AF,'RBF')
    Hout=radbas(H);
elseif strcmp(AF,'tf')
    Hout=tribas(H);
end
%%
Hinv=pinv(Hout); % pinv(A) returns the Moore-Penrose Pseudoinverse of matrix A.
% ELM outputs predicted on the training dataset
%Outputweights=(Hinv)*(xtest); 
OutputweightsA1=(Hinv)*(xtrain(:,1)); % a1
OutputweightsA2=(Hinv)*(xtrain(:,2)); % a2
ModelOutputsA1=Hout*OutputweightsA1; %Salidas de ELM predichas en el conjunto de datos de entrenamiento
ModelOutputsA2=Hout*OutputweightsA2; %Salidas de ELM predichas en el conjunto de datos de entrenamiento
%testing the ELM model
xnew=(xtest-mux)./sigmax;       % test data is normalized
Prod=Inputweights*xnew';
H=Prod+Bias_HL(:,ones(1,size(xnew,1))); 
%% chossing activation function
if strcmp(AF,'tanh')            
    Hout=tanh(H);
elseif strcmp(AF,'sig')
    Hout=1./(1+exp(-H));
elseif strcmp(AF,'sin')
    Hout=sin(H);
elseif strcmp(AF,'cos')
    Hout=cos(H);
elseif strcmp(AF,'RBF')
    Hout=radbas(H);
elseif strcmp(AF,'tf')
    Hout=tribas(H);
end
%% Hout=1./(1+exp(-H));
YpredA1=(Hout')*OutputweightsA1;
YpredA2=(Hout')*OutputweightsA2;
ypredA1=YpredA1.*sigmay+muy;
ypredA2=YpredA2.*sigmay+muy;
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train;        %   Calculate CPU time (seconds) spent for training ELM
fprintf('================================================================================\n')
fprintf('|                                                                               |\n')
fprintf('|                                                                               |\n')
fprintf('|                                                                               |\n')
fprintf('================================================================================\n')
fprintf('Tiempo de EML entrenamiento= %4.4f \n',TrainingTime);
%% PLOT 
% fprintf('================================================================================\n')
% fprintf('|                                      EML                                      |\n')
% fprintf('================================================================================\n')
% R_volt=corr(xtrain(:,1),ypred(:,1));            % correlation coefficient
% fprintf('R(1)= %4.4f \n',R_volt)
% RMSE_volt=sqrt(mean((ypred(:,1)-xtrain(:,1)).^2));
% fprintf('RMSE(1)= %4.4f \n',RMSE_volt)
% MAE_volt=mae(ypred(:,1)-xtrain(:,1));
% fprintf('MAE(1)= %4.4f \n',MAE_volt)

R_A1=corr(YpredA1,xtest(:,1));            % correlation coefficient
R_A2=corr(YpredA2,xtest(:,2));            % correlation coefficient
fprintf('R(A1)= %4.4f \n',R_A1)
fprintf('R(A2)= %4.4f \n',R_A2)
RMSE_A1=sqrt(mean((xtest(:,1)-YpredA1).^2));
RMSE_A2=sqrt(mean((xtest(:,2)-YpredA2).^2));
fprintf('RMSE(A1)= %4.4f \n',RMSE_A1)
fprintf('RMSE(A2)= %4.4f \n',RMSE_A2)
MAE_A1=mae(xtest(:,1)-YpredA1);
MAE_A2=mae(xtest(:,2)-YpredA2);
fprintf('MAE(A1)= %4.4f \n',MAE_A1)
fprintf('MAE(A2)= %4.4f \n',MAE_A2)

%% metrica del error 
MMSE = RMSE_A1/xtest(:,1);

%R_ang=corr(Ypred(:,2),xtest(:,2));            % correlation coefficient
% fprintf('R(2)= %4.4f \n',R_ang)
% RMSE_ang=sqrt(mean((xtest(:,2)-Ypred(:,2)).^2));
% fprintf('RMSE(2)= %4.4f \n',RMSE_ang)
% MAE_ang=mae(xtest(:,2)-Ypred(:,2));
% fprintf('MAE(2)= %4.4f \n',MAE_ang)

% R_ang=corr(xtrain(:,2),ypred(:,2));            % correlation coefficient
% fprintf('R(2)= %4.4f \n',R_ang)
% RMSE_ang=sqrt(mean((ypred(:,2)-xtrain(:,2)).^2));
% fprintf('RMSE(2)= %4.4f \n',RMSE_ang)
% MAE_ang=mae(ypred(:,2)-xtrain(:,2));
% fprintf('MAE(2)= %4.4f \n',MAE_ang)

if p==1
figure(1)
scatter(YpredA1,xtest(:,1),'filled')
xlabel('Potencia actual')
ylabel('Potencia predicho')
title('Potencia Activa [p.u]')

figure(2); 
plot(abs(xtrain(:,1)-ModelOutputsA1)); 
xlabel('No datos')
ylabel('Error')
title('Error absoluto de Potencia Activa')
%%
figure(3)
scatter(YpredA2,xtest(:,2),'filled')
xlabel('Potencia actual')
ylabel('Potencia predicho')
title('Potencia Reactiva [p.u]')

figure(4); 
plot(abs(xtrain(:,2)-ModelOutputsA2)); 
xlabel('No datos')
ylabel('Error')
title('Error absoluto de Potencia Reactiva')
elseif p==2
figure(1)
scatter(YpredA1,xtest(:,1),'filled')
xlabel('Voltaje actual')
ylabel('Voltaje predicho')
title('Voltage 1 [p.u]')

figure(2); 
plot(abs(xtrain(:,1)-ModelOutputsA1)); 
xlabel('No datos')
ylabel('Error')
title('Error absoluto de Voltage')
%%
figure(3)
scatter(YpredA2,xtest(:,2),'filled')
xlabel('Angulo actual')
ylabel('Angulo predicho')
title('Angulo')

figure(4); 
plot(abs(xtrain(:,2)-ModelOutputsA2)); 
xlabel('No datos')
ylabel('Error')
title('Error absoluto de Angulo')
end

% figure(12)
% plot(ytest(:,1))
% hold on
% plot(ypred(:,1))
% legend('Actual data','Predictions')
% xlabel('samples')
% ylabel('mag(pu)')
% title('Voltage')
% 
% figure(13)
% plot(ytest(:,2))
% hold on
% plot(ypred(:,2))
% legend('Actual data','Predictions')
% xlabel('samples')
% ylabel('Ang(deg)')
% title('Angulo')
% 
% figure(14)
% scatter(xtest(:,2)*(180/pi),ypred(:,2)*(180/pi),'filled')
% xlabel('Angulo actual')
% ylabel('angulo predicho AGWN')
% title('Angulo 1 (grados)')

%% distorcion en el case ruidos gaussianos con una media de cero y una desviación estándar del 5%.
%distorcion a la carga 
start_time_test=cputime; % tiempo de prueba

if p==1 %potencia
Pd=caserun.bus(:,3); Qd=caserun.bus(:,4);
%agregamos al case
caserun.bus(:,3)= (imnoise(Pd,'gaussian',0,vargauss))+Pd;
caserun.bus(:,4)= (imnoise(Qd,'gaussian',0,vargauss))+Qd;
copf = runopf(caserun); % ejecutamos el caso con el ruido gaussiano
opfPg=(caserun.bus(:,3))/(base); % tomamos la columba de Pd p.u 
opfQg=(caserun.bus(:,4))/(base); % tomamos la columba de Qd p.u
vargeneral=[opfPg,opfQg];
elseif p==2  %voltaje
Pd=caserun.bus(:,8); Qd=caserun.bus(:,9);
%agregamos al case
caserun.bus(:,8)= (imnoise(Pd,'gaussian',0,vargauss))+Pd;
caserun.bus(:,9)= (imnoise(Qd,'gaussian',0,vargauss))+Qd;
copf = runopf(caserun); % ejecutamos el caso con el ruido gaussiano
magnitudV = copf.bus(:,8); % tomamos la magnitud del voltaje p.u
angulo = (copf.bus(:,9))*(pi/180); % tomamos el angulo deg
opfPg=(caserun.bus(:,3))/(base); % tomamos la columba de Pd p.u 
opfQg=(caserun.bus(:,4))/(base); % tomamos la columba de Qd p.u
vargeneral=[opfPg,opfQg];
elseif p==3 %lambda
Pd=caserun.bus(:,14); Qd=caserun.bus(:,15);
%agregamos al case
caserun.bus(:,14)= (imnoise(Pd,'gaussian',0,vargauss))+Pd;
caserun.bus(:,15)= (imnoise(Qd,'gaussian',0,vargauss))+Qd;
copf = runopf(caserun); % ejecutamos el caso con el ruido gaussiano
LambdaP=copf.bus(:,14);%Lambda($/MVA-hr) Potencia activa
LambdaQ=copf.bus(:,15);%Lambda($/MVA-hr) Potencia reactiva
opfPg=(caserun.bus(:,3))/(base); % tomamos la columba de Pd p.u 
opfQg=(caserun.bus(:,4))/(base); % tomamos la columba de Qd p.u
vargeneral=[opfPg,opfQg];
end
xtrain1=vargeneral(1:Ns,:); % input  
[xtrain1,muy1,sigmay1] = zscore(xtrain1);
Prod=xtrain1*Inputweights';
H=Prod+Biasmatrix'; 
%% chossing activation function
if strcmp(AF,'tanh')            
    Hout=tanh(H);
elseif strcmp(AF,'sig')
    Hout=1./(1+exp(-H));
elseif strcmp(AF,'sin')
    Hout=sin(H);
elseif strcmp(AF,'cos')
    Hout=cos(H);
elseif strcmp(AF,'RBF')
    Hout=radbas(H);
elseif strcmp(AF,'tf')
    Hout=tribas(H);
end
Hinv=pinv(Hout); % pinv(A) returns the Moore-Penrose Pseudoinverse of matrix A.
%Outputweights1=(Hinv)*ytrain; % Voltaje
YpredA11=(Hout)*OutputweightsA1;
YpredA22=(Hout)*OutputweightsA2;
ypredA1=YpredA11.*sigmay1+muy1;
ypredA2=YpredA22.*sigmay1+muy1;
ypredA1=ypredA1(1:length(xtest(:,1)),:);
ypredA2=ypredA2(1:length(xtest(:,2)),:);
end_time_test=cputime;
TestTime=end_time_test-start_time_test;        %   Calculate CPU time (seconds) spent for training ELM

fprintf('================================================================================\n')
fprintf('|                                      EML                                      |\n')
fprintf('================================================================================\n')

R_A1=corr(YpredA1,xtest(:,1));            % correlation coefficient
R_A2=corr(YpredA2,xtest(:,2));            % correlation coefficient
fprintf('R(A1)= %4.4f \n',R_A1)
fprintf('R(A2)= %4.4f \n',R_A2)
RMSE_A1=sqrt(mean((xtest(:,1)-YpredA1).^2));
RMSE_A2=sqrt(mean((xtest(:,2)-YpredA2).^2));
fprintf('RMSE(A1)= %4.4f \n',RMSE_A1)
fprintf('RMSE(A2)= %4.4f \n',RMSE_A2)
MAE_A1=mae(xtest(:,1)-YpredA1);
MAE_A2=mae(xtest(:,2)-YpredA2);
fprintf('MAE(A1)= %4.4f \n',MAE_A1)
fprintf('MAE(A2)= %4.4f \n',MAE_A2)

fprintf('================================================================================\n')
fprintf('|                              Gaussian Noises                                  |\n')
fprintf('================================================================================\n')
fprintf('Tiempo de EML prueba= %4.4f \n',TestTime);

R_A1=corr(ypredA1(:,1),xtest(:,1));            % correlation coefficient
R_A2=corr(ypredA2(:,2),xtest(:,2));            % correlation coefficient
fprintf('R1(A1)= %4.4f \n',R_A1)
fprintf('R1(A2)= %4.4f \n',R_A2)
RMSE_A1=sqrt(mean((xtest(:,1)-ypredA1(:,1)).^2));
RMSE_A2=sqrt(mean((xtest(:,2)-ypredA2(:,2)).^2));
fprintf('RMSE(A1)= %4.4f \n',RMSE_A1)
fprintf('RMSE(A2)= %4.4f \n',RMSE_A2)
MAE_A1=mae(xtest(:,1)-ypredA1(:,1));
MAE_A2=mae(xtest(:,2)-ypredA2(:,2));
fprintf('MAE(A1)= %4.4f \n',MAE_A1)
fprintf('MAE(A2)= %4.4f \n',MAE_A2)

fprintf('================================================================================\n')
fprintf('|                                   THE END                                     |\n')
fprintf('================================================================================\n')

% R1_A1=corr(xtest(:,1),ypredA1);            % correlation coefficient
% R1_A2=corr(xtest(:,2),ypredA2);            % correlation coefficient
% fprintf('R1(A1) AGWN= %4.4f \n',R1_A1)
% fprintf('R1(A2) AGWN= %4.4f \n',R1_A2)
% RMSE1_A1=sqrt(mean((ypredA1-xtest(:,1)).^2));
% RMSE1_A2=sqrt(mean((ypredA2-xtest(:,2)).^2));
% fprintf('RMSE1(A1) AGWN= %4.4f \n',RMSE1_A1)
% fprintf('RMSE1(A2) AGWN= %4.4f \n',RMSE1_A2)
% MAE1_A1=mae(ypredA1-xtest(:,1));
% MAE1_A2=mae(ypredA2-xtest(:,2));
% fprintf('MAE1(A1)= %4.4f \n',MAE1_A1)
% fprintf('MAE1(A2)= %4.4f \n',MAE1_A2)



if p==1
figure(11)
scatter(xtest(:,1),ypredA1(:,1),'filled')
xlabel('Potencia actual')
ylabel('Potencia predicho')
title('Potencia Activa [p.u]')
%%
figure(12)
scatter(xtest(:,2),ypredA2(:,2),'filled')
xlabel('Potencia actual')
ylabel('Potencia predicho')
title('Potencia Reactiva [p.u]')

% figure(13); 
% plot(abs(xtrain(:,1)-ModelOutputsA1)); 
% xlabel('No datos')
% ylabel('Error')
% title('Error absoluto de Potencia Activa AGWN')
elseif p==2
figure(11)
scatter(xtest(:,1),ypredA1(:,1),'filled')
xlabel('Voltaje actual')
ylabel('Voltaje predicho')
title('Voltage 1 [p.u]')
%%
figure(12)
scatter(xtest(:,2),ypredA2(:,2),'filled')
xlabel('Angulo actual')
ylabel('Angulo predicho')
title('Angulo')
end

% figure(21)
% scatter(xtest(:,1),ypred1(:,1),'filled')
% xlabel('Voltaje actual ')
% ylabel('Voltaje predicho AGWN')
% title('Voltage 2 [p.u]')

% figure(22)
% plot(ytest(:,1))
% hold on
% plot(ypred(:,1))
% hold on
% plot(ypred1(:,1))
% legend('Actual data','Predictions','Predictions AWGN')
% xlabel('samples')
% ylabel('mag(pu)')
% title('Voltage')
% 
% figure(23)
% plot(ytest(:,2))
% hold on
% plot(ypred(:,2))
% hold on
% plot(ypred1(:,2))
% legend('Actual data','Predictions','Predictions AWGN')
% xlabel('samples')
% ylabel('Ang(deg)')
% title('Angulo')
% 
% figure(24)
% scatter(xtest(:,2)*(180/pi),ypred1(:,2)*(180/pi),'filled')
% xlabel('Angulo actual')
% ylabel('angulo predicho AGWN')
% title('Angulo 2 (grados)')

% fprintf('================================================================================\n')
% fprintf('|                                   THE END                                     |\n')
% fprintf('================================================================================\n')
