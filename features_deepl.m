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
sxmm=[];
s=[];
for i=1:4
    ds = imageDatastore(locations(i));
    for ii=1:size(ds.Files,1)

        x=readimage(ds,ii);
        x_sharp=imsharpen(x,"Radius",2,"Amount",2,"Threshold",0.1);
        [cA,cH,cV,cD] = dwt2(x_sharp,'haar');
        x_r=x_sharp(:,:,1);
        x_g=x_sharp(:,:,2);
        x_fr=fft2(x_r);
        x_fg=fft2(x_g);
        x_fr= imresize(x_fr,[64 64]);
        x_fg = imresize(x_fg,[64 64]);
        x_fr=reshape(x_fr,1,[]);
        x_fg=reshape(x_fg,1,[]);
        
        cAwr=cA(:,:,1);
        cAwg=cA(:,:,2);
        cAwr= imresize(cAwr,[64 64]);
        cAwg = imresize(cAwg,[64 64]);
        cAwr=reshape(cAwr,1,[]);
        cAwg=reshape(cAwg,1,[]);
        x_all=[x_fr x_fg cAwr cAwg];
        sxm(ii,:)=x_all;
    end
    
  sxmm=[sxmm;sxm];
  sxm=[];
end
 net = trainNetwork(feature_1,lgraph,'lgraph');