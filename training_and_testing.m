% training:
%
% Kadid10k
% PLCC  0.9424
% SROCC 0.9496
% KROCC 0.8036
%
% testing:
% LIVE
% PLCC  0.9560
% SROCC 0.9627
% KROCC 0.8280
%
% CSIQ
% PLCC  0.9305
% SROCC 0.9393
% KROCC 0.7812
%
% TID2013
% PLCC  0.8884
% SROCC 0.8767
% KROCC 0.6899

clc
clear
close all

% load data  and MOS (1482-th column) normalization

load("kadid.mat");
Mkadid = 5;
mkadid = 1;
kadid(:,1482) = (kadid(:,1482)-mkadid)/(Mkadid-mkadid);

load("live.mat");
load("csiq.mat");
load("tid.mat");

Mlive = 120;
mlive = 0;


Mcsiq = 1.03;
mcsiq = -0.05;

Mtid = 7.2;
mtid = 0;

% data preprocessing (features normalization)

data = [kadid;live;csiq;tid];
temp = data(:,1482);
M = max(data,[],1);
M = repmat(M,size(data,1),1);
m = min(data,[],1);
m = repmat(m,size(data,1),1);
data = (data - m)./(M-m);
data(:,1482) = temp;
kadid = data(1:10125,:);
live =  data(10126:10904,:);
csiq= data(10905:11770,:);
tid = data(11771:14770,:);


% training
tic

train = kadid;
A = [ones(size(train,1),1),train(:,1:1481)];
lambda =75;
I = eye(size(A,2));
I(1,1) = 0;
b = train(:,1482);
H = (A'*A) +lambda*I ;
H = (H+H')/2;
f = A'*b;
w = H\f;

toc

figure(1)
X = [ones(size(kadid,1),1),kadid(:,1:1481)];
Y = kadid(:,1482);
Y = (Mkadid-mkadid)*Y+mkadid;
YPred = X*w;
YPred = (Mkadid-mkadid)*YPred+mkadid;
plot(Y,YPred,'o');
[cf, ~]=L5P(YPred,Y);
hold on
A = cf.A;B = cf.B;C = cf.C;D = cf.D;E = cf.E;
PLCC1 = corr(Y,YPred,'type','Pearson');
SROCC1 = corr(Y,YPred,'type','Spearman');
KROCC1 = corr(Y,YPred,'type','Kendall');
fprintf("Kadid10k\nPLCC\t%6.4f\nSROCC\t%6.4f\nKROCC\t%6.4f\n\n",PLCC1,SROCC1,KROCC1);
x = linspace(mkadid,Mkadid,1000);
[cf, ~]=L5P(Y,YPred);
A = cf.A;B = cf.B;C = cf.C;D = cf.D;E = cf.E;
y = D+(A-D)./((1+(x./C).^B).^E);
plot(x,y,'-','LineWidth',2)
ylabel({'Predicted Quality Score'},'FontName','Times New Roman','Interpreter','latex');
xlabel({'MOS'},'Interpreter','latex');
grid on

figure(2)
X = [ones(size(live,1),1),live(:,1:1481)];
Y = 114-live(:,1482);
YPred = X*w;
plot(Y,X*w,'o');
hold on
PLCC2 = corr(Y,YPred,'type','Pearson');
SROCC2 = corr(Y,YPred,'type','Spearman');
KROCC2 = corr(Y,YPred,'type','Kendall');
fprintf("LIVE\nPLCC\t%6.4f\nSROCC\t%6.4f\nKROCC\t%6.4f\n\n",PLCC2,SROCC2,KROCC2);
x = linspace(mlive,Mlive,1000);
[cf, ~]=L5P(Y,YPred+0.6);
A = cf.A;B = cf.B;C = cf.C;D = cf.D;E = cf.E;
y = D+(A-D)./((1+(x./C).^B).^E)-0.6;
plot(x,y,'-','LineWidth',2)
ylabel({'Predicted Quality Score'},'FontName','Times New Roman','Interpreter','latex');
xlabel({'MOS'},'Interpreter','latex');
grid on

figure(3)
X = [ones(size(csiq,1),1),csiq(:,1:1481)];
Y = csiq(:,1482);
YPred = X*w;
plot(Y,X*w,'o');
hold on
PLCC3 = corr(Y,YPred,'type','Pearson');
SROCC3 = corr(Y,YPred,'type','Spearman');
KROCC3 = corr(Y,YPred,'type','Kendall');
fprintf("CSIQ\nPLCC\t%6.4f\nSROCC\t%6.4f\nKROCC\t%6.4f\n\n",PLCC3,SROCC3,KROCC3);
x = linspace(mcsiq,Mcsiq,1000);
[cf, ~]=L5P(Y,YPred);
A = cf.A;B = cf.B;C = cf.C;D = cf.D;E = cf.E;
y = D+(A-D)./((1+(x./C).^B).^E);
plot(x,y,'-','LineWidth',2)
ylabel({'Predicted Quality Score'},'FontName','Times New Roman','Interpreter','latex');
xlabel({'MOS'},'Interpreter','latex');
grid on

figure(4)
X = [ones(size(tid,1),1),tid(:,1:1481)];
Y = tid(:,1482);
YPred = X*w;
plot(Y,X*w,'o');
hold on
PLCC4 = corr(Y,YPred,'type','Pearson');
SROCC4 = corr(Y,YPred,'type','Spearman');
KROCC4 = corr(Y,YPred,'type','Kendall');
fprintf("TID2013\nPLCC\t%6.4f\nSROCC\t%6.4f\nKROCC\t%6.4f\n\n",PLCC4,SROCC4,KROCC4);
x = linspace(mtid,Mtid,1000);
[cf, ~]=L5P(Y,YPred);
A = cf.A;B = cf.B;C = cf.C;D = cf.D;E = cf.E;
y = D+(A-D)./((1+(x./C).^B).^E);
plot(x,y,'-','LineWidth',2)
ylabel({'Predicted Quality Score'},'FontName','Times New Roman','Interpreter','latex');
xlabel({'MOS'},'Interpreter','latex');
grid on

set(gcf,'windowstyle','normal');
set(gcf,'Position',[100 100 650 550])