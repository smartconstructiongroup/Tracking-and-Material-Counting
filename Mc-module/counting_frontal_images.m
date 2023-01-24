%This section is related to frontal view
clc;
clear all;
close all;

Directory = '1\';
% Read images from Images folder
Imgs = dir(fullfile(Directory,'*.jpg'));

% fileID = fopen('1\foutput.txt','w');
% fileID2 = fopen('1\fgroundtruth.txt','w');
for j=1:length(Imgs) 
im = imread(fullfile(Directory,Imgs(j).name)); % Read image

label=1; 

if(label==1)
    I=(im);
    T = adaptthresh(I, 0.6);
    BW = imbinarize(I,T);
    
 
    figure;
    imshowpair(I, BW, 'montage')

    se = strel('line',10,180);
    erodedBW = imerode(BW,se);
    se = strel('line',25,0);
    erodedBW2 = imerode(erodedBW,se);
    figure
    imshowpair(BW, erodedBW2, 'montage')

    se1 = strel('disk',3);
    afterOpening = imopen(erodedBW2,se1);
    figure, imshowpair(erodedBW2,afterOpening, 'montage')
    BW2 = bwareaopen(afterOpening, 300);

    figure, imshowpair(afterOpening,BW2, 'montage')
    temp=zeros(300,300);
    temp(15:end-10,40:end-60)=BW2(15:end-10,40:end-60);
    figure, imshowpair(BW2,temp, 'montage')
    
    se = strel('line',20,180);
    temp2 = imdilate(temp,se);
    figure, imshowpair(temp,temp2, 'montage');
    
    L1=60;
    row_start=-1;
    t=1;
    while(t<round(size(temp2,1)/2) && row_start<0 )
        for s=1:size(temp2,1)-L1
            p=temp2(t,s:s+L1);
        if(temp2(t,s:s+L1)==1)
            row_start=t;
            pp=temp2(t,s:s+L1);
            continue;
        end
        end
        t=t+1;
    end
    
    BW3 = bwmorph(temp2,'skel',Inf);
    
    
    figure, imshowpair(temp2,BW3, 'montage');
    
    
       %% Detect Lines
    % Perform Hough Transform
    [H,T,R] = hough(BW3);
%     [H,T,R] = hough(BW);

% %     Identify Peaks in Hough Transform
% % % % % % % % % % % % % % % % % % % % % % % 

    hPeaks =  houghpeaks(H,20,'NHoodSize',[35 35]);

    % Extract lines from hough transform and peaks
    hLines = houghlines(BW3,T,R,hPeaks,...
            'FillGap',80,'MinLength',90);
        
   [linePos,markerPos] = getVizPosArray(hLines);

   lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',5);

    figure
%     imshowpair(im, RGB, 'montage'),title('The number of Palette is');
    imshowpair(im, lineFrame, 'montage');  
    
        
    %% View results
    % Overlay lines
    [linePos,markerPos] = getVizPosArray(hLines);

    endColumn=linePos(:,3);
    maxend=max(endColumn);
    linePos(:,3)=maxend;
    
    startColumn=linePos(:,1);
    maxstart=max(startColumn);
    maxmin=min(startColumn);
    linePos(:,1)=maxstart-round((maxstart-maxmin)/2);
    
    
    sorted1=sort(linePos(:,2));
    sorted2=sort(linePos(:,4));
    linePos(:,2)=sorted1;
    linePos(:,4)=sorted2;
    
 

   lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',5);
    figure
    imshowpair(im, lineFrame, 'montage');  
%     imshowpair(im, RGB, 'montage'),title('The number of Palette is');
 
    
    a=zeros(size(linePos,1),1);
    b=zeros(size(linePos,1),1);
    for t=1:size(linePos,1)
        coefficients = polyfit([linePos(t,1), linePos(t,3)], [linePos(t,2), linePos(t,4)], 1);
        a(t) = coefficients (1);
        b(t) = coefficients (2);
    end
  
    a_p=mean(a(:));
    
   for t=1:size(linePos,1)
        a(t) = a_p;
        linePos(t,2)=round(linePos(t,1)*a(t)+b(t));
        linePos(t,4)=round(linePos(t,3)*a(t)+b(t));
   end
    
[sorted_b, Indexes]=sort(b);

    b=sorted_b;
   
   
    endColumn=linePos(:,3);
    maxend=min(endColumn);
    linePos(:,3)=maxend;
%     
    startColumn=linePos(:,1);
    maxstart=min(startColumn);
    linePos(:,1)=maxstart;
    
    
    
    lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',5);

    figure
    imshowpair(im, lineFrame, 'montage');  
%     imshowpair(im, RGB, 'montage'),title('The number of Palette is');

%%

max1=0;
min1=300;
idx=-1;

max2=0;
min2=300;
idx2=-1;

sorted1=sort(linePos(:,2));
sorted2=sort(linePos(:,4));
linePos(:,2)=sorted1;
linePos(:,4)=sorted2;

if(max(linePos(:,2))<265 | max(linePos(:,4))<265)
    
    newLinePos=zeros(size(linePos,1)+1,4);
    newLinePos(1:end-1,:)=linePos;
    
    newLinePos(end,:)=linePos(end,:);
    newLinePos(end,2)=linePos(end,2)+45;
    newLinePos(end,4)=linePos(end,4)+45;
    linePos=newLinePos;
end


   lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',5);

    figure
    imshowpair(im, lineFrame, 'montage');  
%     imshowpair(im, RGB, 'montage'),title('The number of Palette is');

points=abs(min(linePos(:,4))-row_start);
if(min(linePos(:,2))>60 && min(linePos(:,4))>60 && points>25)
    
    newLinePos=zeros(size(linePos,1)+1,4);
    newLinePos(2:end,:)=linePos;
    
    newLinePos(1,:)=linePos(1,:);
    newLinePos(1,2)=row_start+10;
    newLinePos(1,4)=row_start+10;
    linePos=newLinePos;
end

   lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',5);
    figure
    imshowpair(im, lineFrame, 'montage');  
%     imshowpair(im, RGB, 'montage'),title('The number of Palette is');

for k=1:size(linePos,1)-1
    temp=abs(linePos(k,2)-linePos(k+1,2));
    temp2=abs(linePos(k,4)-linePos(k+1,4));
    if(temp>max1)
        max1=temp;
        idx=k;
    end
    if(temp<min1)
        min1=temp;
    end

    if(temp2>max2)
        max2=temp2;
        idx2=k;
    end
    if(temp2<min2)
        min2=temp2;
    end


end

%%



  lineFrame = insertShape(im,'Line',linePos,...
'Color','yellow','LineWidth',5);

figure
imshowpair(im, lineFrame, 'montage');  



min_final=min1;
min1=300;
go=1;

while(go) 
   if (size(linePos,1)<15) 
        max1=0;
        min1=300;
        idx=-1;

        max2=0;
        min2=300;
        idx2=-1;

        sorted1=sort(linePos(:,2));
        sorted2=sort(linePos(:,4));
        linePos(:,2)=sorted1;
        linePos(:,4)=sorted2;
        
        
        for k=1:size(linePos,1)-1
            temp=abs(linePos(k,2)-linePos(k+1,2));
            temp2=abs(linePos(k,4)-linePos(k+1,4));
            if(temp>max1)
                max1=temp;
                idx=k;
            end
            if(temp<min1)
                min1=temp;
            end

            if(temp2>max2)
                max2=temp2;
                idx2=k;
            end
            if(temp2<min2)
                min2=temp2;
            end


        end

        if(idx2>idx)
            idx=idx2;
        end
        addedline=max1/min1;
        newLinePos=zeros(size(linePos,1)+1,4);
        
        
        if((abs(min_final-min1)<7)&& addedline>1.85)
                for y=1:idx
                    newLinePos(y,:)=linePos(y,:);
                end

                newLinePos(idx+1,2)=linePos(idx,2)+min1;
                newLinePos(idx+1,4)=linePos(idx,4)+min2;
                newLinePos(idx+1,1)=linePos(idx,1);
                newLinePos(idx+1,3)=linePos(idx,3);
                
                
                newLinePos(idx+2:end,:)=linePos(idx+1:end,:);
                linePos=newLinePos;
                
        elseif((abs(min_final-min1)>7)&& addedline>1)
%             while(p<addedline)
                for y=1:idx
                    newLinePos(y,:)=linePos(y,:);
                end

                newLinePos(idx+1,2)=linePos(idx,2)+min1+abs(min_final-min1);
                newLinePos(idx+1,4)=linePos(idx,4)+min2+abs(min_final-min1);
                newLinePos(idx+1,1)=linePos(idx,1);
                newLinePos(idx+1,3)=linePos(idx,3);
                
                
                newLinePos(idx+2:end,:)=linePos(idx+1:end,:);
                linePos=newLinePos;       
                
                           
        else
            go=0;
        end 
        
                  lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',5);

                figure
                imshowpair(im, lineFrame, 'montage');  

    else
        break;
    end 
                
end
close all;
   lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',5);

    figure

    imshow(lineFrame);  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     imwrite(RGB,strcat('1\results\',Imgs(j).name));
%     
%     fprintf(fileID,'%s %d\n',Imgs(j).name,size(linePos,1));
%     fprintf(fileID2,'%s %d\n',Imgs(j).name,0);
%     newStr = extractBefore(Imgs(j).name,'.jpg');
%     file_name=strcat(newStr,'.txt');
%     fid=fopen(file_name,'w');
% 
%     for k = 1:size(linePos,1)
% 
%        fprintf(fid, '%f %f %f %f \r\n', [linePos(k,1) linePos(k,2) linePos(k,3) linePos(k,4)]);
% 
%     end
% 
%     fclose(fid);



%    lineFrame = insertShape(im,'Line',linePos,...
%                 'Color','white','LineWidth',5);
%     position =  [150 20];
%     RGB = insertText(lineFrame,position,size(linePos,1),'AnchorPoint','LeftBottom');
%     im = insertText(im,position,size(linePos,1),'AnchorPoint','LeftBottom');
%     figure
%     imshowpair(im, RGB, 'montage'),title('The number of Palette is');


    
end

end
