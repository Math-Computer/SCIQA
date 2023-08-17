clear
clc
close

load("model_parameters.mat"); % the predtrained SGIQA model
params = load('vgg16net_param.mat'); % the pretrained vgg16 parameters
resize_img = 1; % resize the input image  so that the short side of the image is 256

%========================================================================

ref = imread(".\images\1.png"); % reference image

for i = 2:4
    str = num2str(i);
    dist = imread(strcat(".\images\",str,".png"));% distorted image
    sgiqa =SGIQA(ref,dist,params,resize_img,w,M,m);
    fprintf("%d SGIQA = %6.4f\n",i,sgiqa);
end

%========================================================================

function IQA =SGIQA(ref,dist,params,resize_img,w,M,m)

tic
[ref_Stru,ref_Gram] = deep_features(ref,params,resize_img);
[dist_Stru,dist_Gram] = deep_features(dist,params,resize_img);

% distance features between reference and distorted images
dist_ref_features  = final_features(ref_Stru,ref_Gram,dist_Stru,dist_Gram);

dist_ref_features = (dist_ref_features - m)./(M-m);
X = [ones(size(dist_ref_features,1),1),dist_ref_features];
IQA = X*w;

toc
end

function dist_ref_features = final_features(ref_Stru,ref_Gram,dist_Stru,dist_Gram)

dist_ref_features = [];
chns = [3,64,128,256,512,512];

% 1475 (1-1475)
for i = 1:6
    for k = 1:chns(i)
        temp = norm(dist_Stru{i}(:,:,k)-ref_Stru{i}(:,:,k),'fro');
        dist_ref_features = [dist_ref_features temp];
    end
end

% 6 (1476-1481)
for i = 1:6
    temp = norm(dist_Gram{i}-ref_Gram{i},'fro');
    dist_ref_features = [dist_ref_features temp];
end
end

function [Stru,Gram] = deep_features(I,params,resize_img)
features = extract_features(I,params,resize_img);

% extract structure map
Stru = cell(6,1);

% extract texture Gram matrix and statistical features
Gram = cell(6,1);

for i = 1:6

    data = double(extractdata(features{i}));
    
    % extract structure map
    Stru{i} = data;
    
    
    % extract Gram matrix
    G = zeros(size(data,3),size(data,1)*size(data,2));
    for k = 1:size(data,3)
        G(k,:)=reshape(data(:,:,k),1,size(data,1)*size(data,2));
    end
    Gram{i} = G*G';    

end
end

function features = extract_features(I,params,resize_img)
if resize_img && min(size(I,1),size(I,2))>256
    I = imresize(I,256/min(size(I,1),size(I,2)));
end
I = dlarray(double(I)/255,'SSC');

features = cell(6,1);
% stage 0
features{1} = I;
dlX = (I - params.vgg_mean)./params.vgg_std;

% stage 1
weights = dlarray(params.conv1_1_weight);
bias = dlarray(params.conv1_1_bias');
dlY = relu(dlconv(dlX,weights,bias,'Stride',1,'Padding','same'));

weights = dlarray(params.conv1_2_weight);
bias = dlarray(params.conv1_2_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
features{2} = dlY;

% stage 2
weights = dlarray(params.L2pool_1);
dlY = dlconv(dlY.^2,weights,0,'Stride',2,'Padding',[1, 1; 0, 0]);
dlY = sqrt(dlY);
% dlY = avgpool(dlY,2,'Stride',2);

weights = dlarray(params.conv2_1_weight);
bias = dlarray(params.conv2_1_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

weights = dlarray(params.conv2_2_weight);
bias = dlarray(params.conv2_2_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));
features{3} = dlY;

% stage 3
weights = dlarray(params.L2pool_2);
dlY = dlconv(dlY.^2,weights,0,'Stride',2,'Padding',[1, 1; 0, 0]);
dlY = sqrt(dlY);
% dlY = avgpool(dlY,2,'Stride',2);

weights = dlarray(params.conv3_1_weight);
bias = dlarray(params.conv3_1_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

weights = dlarray(params.conv3_2_weight);
bias = dlarray(params.conv3_2_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

weights = dlarray(params.conv3_3_weight);
bias = dlarray(params.conv3_3_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

features{4} = dlY;

% stage 4
weights = dlarray(params.L2pool_3);
dlY = dlconv(dlY.^2,weights,0,'Stride',2,'Padding',[1, 1; 0, 0]);
dlY = sqrt(dlY);
% dlY = avgpool(dlY,2,'Stride',2);

weights = dlarray(params.conv4_1_weight);
bias = dlarray(params.conv4_1_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

weights = dlarray(params.conv4_2_weight);
bias = dlarray(params.conv4_2_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

weights = dlarray(params.conv4_3_weight);
bias = dlarray(params.conv4_3_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

features{5} = dlY;

% stage 5
weights = dlarray(params.L2pool_4);
dlY = dlconv(dlY.^2,weights,0,'Stride',2,'Padding',[1, 1; 0, 0]);
dlY = sqrt(dlY);
% dlY = avgpool(dlY,2,'Stride',2);

weights = dlarray(params.conv5_1_weight);
bias = dlarray(params.conv5_1_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

weights = dlarray(params.conv5_2_weight);
bias = dlarray(params.conv5_2_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

weights = dlarray(params.conv5_3_weight);
bias = dlarray(params.conv5_3_bias');
dlY = relu(dlconv(dlY,weights,bias,'Stride',1,'Padding','same'));

features{6} = dlY;
end