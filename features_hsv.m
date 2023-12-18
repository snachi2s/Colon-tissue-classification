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

%         x=imsharpen(x,"Radius",2);
        x1=imadjust(rgb2hsv(x),[0.6 0.9],[0.6 0.98]);
       
        BW1 =edge(x1(:,:,1),"canny",0.15);

        se90 = strel("line",3,0);
        s0=strel('disk',1,0);
        imshow(BW1);
        BWsdil1=imopen(BWsdil1,1);
        
         BWsdil1=bwareafilt(BWsdil1,[30 inf]);

        Bdwfill1=imfill(BWsdil1,"holes");
        imshowpair(x,Bdwfill1,"Montage");
        s= regionprops(Bdwfill1,'Circularity');
        l= regionprops(Bdwfill1,'Area');
        k = struct2array(l)';
        sx=k;
        sxm(ii,1)=mean(k);
        
        hold off
     

    end


    cmp(1:size(sxm,1),i)= sxm';
    sxm=[];
end

figure
hold on
plot(cmp(:,1),'.',Color=[0 0 0]);
plot(cmp(:,2),'.',Color='r');
plot(cmp(:,3),'.',Color='y');
plot(cmp(:,4),'.',Color='g');

