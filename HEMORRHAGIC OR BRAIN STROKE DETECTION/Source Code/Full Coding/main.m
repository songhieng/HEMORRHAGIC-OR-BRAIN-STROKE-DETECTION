%% Brain Stroke Detection based on Two Segmentation Model

clear all;
close all;
clc;

%% Read Test Image 

[filename,pathname] = uigetfile('*.jpg;*.tif;*.png;*.jpeg;*.bmp;*.pgm;*.gif','pick an imgae');
file = fullfile(pathname,filename);

   img = (imread(file));

   figure,
   imshow(img);
   title('Test Image');
   

%% Preprocessing

if size(img,3) == 3
    img=rgb2gray(img);
end

J = imnoise(img,'salt & pepper',0.02);
figure,
imshow(J);
title('Noise Image');


J1=medfilt2(J,[3 3]);
figure,
imshow(J1);
title('Filtered Image');


%% Segmentation - Morphological Operation 

f=J1;
[x,y]=size(f);
p=zeros(x,y);p2=zeros(x,y);p12=zeros(x,y);p13=zeros(x,y);

% Dilation

w=[1 1 1; 1 1 1; 1 1 1];
for s=2:x-2
    for t=2:y-2
        w1=[f(s-1,t-1)*w(1) f(s-1,t)*w(2) f(s-1,t+1)*w(3) f(s,t-1)*w(4) f(s,t)*w(5) f(s,t+1)*w(6) f(s+1,t-1)*w(7) f(s+1,t)*w(8) f(s+1,t+1)*w(9)];
        p(s,t)=max(w1);
    end
end
figure,
imshow(uint8(p));
title('Dilated Image');

% Erosion

w=[1 1 1; 1 1 1; 1 1 1];
for s=2:x-1
    for t=2:y-1
        w12=[f(s-1,t-1)*w(1) f(s-1,t)*w(2) f(s-1,t+1)*w(3) f(s,t-1)*w(4) f(s,t)*w(5) f(s,t+1)*w(6) f(s+1,t-1)*w(7) f(s+1,t)*w(8) f(s+1,t+1)*w(9)];
        p1(s,t)=min(w12);
    end
end
figure,
imshow(uint8(p1));
title('Eroded Image');

% Opening of image

[m,n]=size(p);
w=[1 1 1; 1 1 1; 1 1 1];
for s=2:m-1
    for t=2:n-1
        w13=[p(s-1,t-1)*w(1) p(s-1,t)*w(2) p(s-1,t+1)*w(3) p(s,t-1)*w(4) p(s,t)*w(5) p(s,t+1)*w(6) p(s+1,t-1)*w(7) p(s+1,t)*w(8) p(s+1,t+1)*w(9)];
        p12(s,t)=min(w13);
    end
end
figure,
imshow(uint8(p12));
title('Opening of Image');

% Closing of image

[r,c]=size(p1);
w=[1 1 1; 1 1 1; 1 1 1];
for s=2:r-1
    for t=2:c-1
        w14=[p1(s-1,t-1)*w(1) p1(s-1,t)*w(2) p1(s-1,t+1)*w(3) p1(s,t-1)*w(4) p1(s,t)*w(5) p1(s,t+1)*w(6) p1(s+1,t-1)*w(7) p1(s+1,t)*w(8) p1(s+1,t+1)*w(9)];
        p13(s,t)=min(w14);
    end
end
figure,
imshow(uint8(p13));
title('Closing of Image');


% Thresholding Method

BS = im2bw(uint8(p13),0.6);     % make binary, and remove noise
figure,
imshow(BS);
title('Initial Segmented Image');

FBS = bwareaopen(BS, 225);              % Remove small object
figure,
imshow(FBS);
title('Segmented Image using Thresholding Method');

%% Segmentation using Seeded Region Growing


img=im2double(img);
x=198; y=359;
img1 = imadjust(img);
SR = region((img1),x,y,0.7);
SR=~SR;
figure,imshow(SR);
title('Segmented Image using SRG');

BWfinal= (double(FBS).*double(SR)) ;
figure,imshow(BWfinal);
title('Final Segmented Image');


%% Feature Extraction


if sum(BWfinal) == 0
    C1 =0;
C2 =0;
C3 =0;
C4 =0;
C5 =0;

    else


stats  = regionprops(BWfinal,'Area','Eccentricity','Solidity','MajorAxisLength','Perimeter');

C1 =stats.Area;
fprintf('AREA \n');
disp(C1);

C2 =stats.Eccentricity;
fprintf('ECCENTRICITY \n');
disp(C2);

C3 =stats.Solidity;
fprintf('SOLIDITY \n');
disp(C3);

C4 =stats.MajorAxisLength;
fprintf('MAJORAXISLENGTH \n');
disp(C4);

C5 =stats.Perimeter;
fprintf('PERIMETER \n');
disp(C5);

end
Feature=[C1 C2 C3 C4 C5];

%% Classification 

%% Classification Using SVM

 load train.mat
 
 xdata = unnamed;
 group = unnamed1;
 svmStruct1 = svmtrain(xdata,group,'kernel_function', 'linear');
 SVM_CLASS = svmclassify(svmStruct1,Feature,'showplot',false);
 
FSR=SVM_CLASS;

if FSR == 1
  msgbox('Abnormal','Result');
    
else
  msgbox('Normal','Result');
   
end


