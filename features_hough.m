%  folder in which your images exists
locations = [("F:\Colon_Classifier\data\data\Adenoma"); ...
    ("F:\Colon_Classifier\data\data\Adenocarcinoma"); ...
    ("F:\Colon_Classifier\data\data\Normal_Tissue"); ...
    ("F:\Colon_Classifier\data\data\Serrated_Lesion")]; 


hold on
pcn = readtable('F:\Colon_Classifier\Colon-tissue-classification\ism_project_2023\ism_project_2023\train.csv');
pcv= readmatrix('F:\Colon_Classifier\Colon-tissue-classification\ism_project_2023\ism_project_2023\train.csv');
pcv(:,1)=[];
cmp=zeros(500);
sxm=[];
for i=1:4
    ds = imageDatastore(locations(i));
    for ii=1:size(ds.Files,1)

        x=readimage(ds,ii);

        % x=imread(string(pcn.name(i))+'.jpg');
       
%         x=imsharpen(x,"Radius",2);
        x1=imadjust(x,[0 0.6],[0.6 0.98]);
        x2=imadjust(x,[0.6 1]);



         BW1 =edge(BW1,"canny",0.5);
        x=createMaskLab(x);
%      x=imclearborder(x,1);
        BW1 =edge(x,"canny",0.15);

        BW2 =edge(x,"canny",0.1);

        se90 = strel("line",3,0);
        s0=strel('disk',1,0);
        BWsdil1=imopen(BW1,1);
        BWsdil1=imopen(BWsdil1,1);
         BWsdil1=bwareafilt(BWsdil1,[30 30000]);

        Bdwfill1=imfill(BWsdil1,"holes");

        r=imfindcircles(Bdwfill1,[60 200]);
          imshowpair(Bdwfill1,x,'Montage');
        
        sxm(ii,1)=mean(r);
     
        hold off
        % plot(sxm,'.');

    end


    cmp(1:size(sxm,1),i)= sxm';
    sxm=[];
end

figure
hold on
plot(fft(cmp(:,1)),'.',Color=[0 0 0]);
plot(fft(cmp(:,2)),'.',Color='r');
plot(fft(cmp(:,3)),'.',Color='y');
plot(fft(cmp(:,4)),'.',Color='g');
writematrix(cmp(:,1),'cool.csv','Delimiter',',')

% train svm

% cool=fitcecoc(sxm,pcv);