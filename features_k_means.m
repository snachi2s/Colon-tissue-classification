%  folder in which your images exists
locations = [("F:\Colon_Classifier\data\data\Adenoma"); ...
    ("F:\Colon_Classifier\data\data\Adenocarcinoma"); ...
    ("F:\Colon_Classifier\data\data\Normal_Tissue"); ...
    ("F:\Colon_Classifier\data\data\Serrated_Lesion")];

sxm=[];

for i=1:4
    ds = imageDatastore(locations(i));
    for ii=1:size(ds.Files,1)

        x=readimage(ds,ii);

        % x=imread(string(pcn.name(i))+'.jpg');


%         x=imsharpen(x,"Radius",2);
        x1=imadjust(x,[0 0.6],[0.6 0.98]);
        x2=imadjust(x,[0.6 1]);


        lab_he = rgb2lab(x1);
        % To segment the image using only color information, limit the image to the a* and b* values in lab_he. Convert the image to data type single for use with imsegkmeans. Use the imsegkmeans function to segment the image into three regions.

        ab = lab_he(:,:,2:3);
        ab = im2single(ab);
        numColors = 2;
        L2 = imsegkmeans(ab,numColors,"NumAttempts",10);
        % Display the label image as an overlay on the original image. The label image separates the white, blue-purple, and pink stained tissue regions.

        B2 = labeloverlay(x1,L2);
        % imshowpair(B2,x1,'Montage');
        title("K-Means")


        BW1 =edge(L2,"canny",0.5);

        x=createMaskLab(x);

        % This founction cleans up the borders of the canny detection
        % algorithm, but can also remove major parts of the image
        % x=imclearborder(x,1);

        %Structuring element
        se90 = strel("line",3,0);
        s0=strel('disk',1,0);

        %dilate and then erode the image
        BWsdil1=imopen(BW1,1);
        BWsdil1=imopen(BWsdil1,1);

        % Filter small and very large blobs
        BWsdil1=bwareafilt(BWsdil1,[30 30000]);

        %Fill the holes to form blobs
        Bdwfill1=imfill(BWsdil1,"holes");

        %Matlabs internal blob detection, the Area and Circularity are
        %important
        imshowpair(Bdwfill1,x,'Montage');
        s= regionprops(Bdwfill1,'Circularity');
        l= regionprops(Bdwfill1,'Area');
        k = struct2array(l)';
        sx=k;

        % Select staticstical features from the areas and circualarities
%         sxm(ii,1)=var(k);
        sxm(ii,1)=mean(k);

        hold off


    end

    %feature vector
    cmp(1:size(sxm,1),i)= sxm';
    sxm=[];
end

figure
hold on
plot(fft(cmp(:,1)),'.',Color=[0 0 0]);
plot(fft(cmp(:,2)),'.',Color='r');
plot(fft(cmp(:,3)),'.',Color='y');
plot(fft(cmp(:,4)),'.',Color='g');