%This section is related to counting images for right view
clc;
clear all;
close all;

Directory = '2\';
% Read images from Images folder
Imgs = dir(fullfile(Directory,'*.jpg'));

% fileID = fopen('1\routput.txt','w');
% fileID2 = fopen('1\rgroundtruth.txt','w');
for j=1:length(Imgs)

im = imread(fullfile(Directory,Imgs(j).name)); % Read image

label=2; 

if(label==2)
    I=im;
    
    
    
    T = adaptthresh(I, 0.6);
    BW = imbinarize(I,T);
    figure;
    imshowpair(I, BW, 'montage')

    temp=zeros(300,300);
    temp(10:end,60:end-60)=BW(10:end,60:end-60);
    figure;
    imshowpair(BW, temp, 'montage')
    
    se = strel('line',12,-35);
    erodedBW = imerode(temp,se);
    figure,imshowpair(temp, erodedBW, 'montage')
    
    BW2=erodedBW;
    BW3 = bwmorph(BW2,'skel',Inf);
    figure, imshowpair(BW2,BW3, 'montage')

    [H,T,R] = hough(BW3);
    

    figure;
    imshow(imadjust(mat2gray(H)),'XData',T,'YData',R,...
          'InitialMagnification','fit');
    xlabel('\theta'), ylabel('\rho');
    axis on, axis normal, hold on;
    colormap(gca,hot);
    
    % Identify Peaks in Hough Transform
    hPeaks =  houghpeaks(H,20,'NHoodSize',[35 35]);

    % Extract lines from hough transform and peaks
    hLines = houghlines(BW3,T,R,hPeaks,...
            'FillGap',80,'MinLength',90);
        

    [linePos,markerPos] = getVizPosArray(hLines);
   
    
   lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',5);

    figure
    imshowpair(im, lineFrame, 'montage'); 
    title('applying morphology')

        [linePos1,markerPos1] = getVizPosArray(hLines1);
   
    
   lineFrame1 = insertShape(im,'Line',linePos1,...
                'Color','yellow','LineWidth',5);

    figure
    imshowpair(im, lineFrame1, 'montage'); 
    title('without applying morphology')
    
    a=zeros(size(linePos,1),1);
    b=zeros(size(linePos,1),1);
    
    for t=1:size(linePos,1)
        coefficients = polyfit([linePos(t,1), linePos(t,3)], [linePos(t,2), linePos(t,4)], 1);
        a(t) = coefficients (1);
        b(t) = coefficients (2);
    end
  
    delete=0;
    v=find((b<-40)==1);
  
    if(size(v,1)>0 | (size(a,1)==1 && a(1)<0))
        b=[295.6260,230.52,173.9412,206.8865,134.4157]';
        a=[0.0153,-0.0200,0.0471,0.0922,0.3072]';
        linePos=zeros(size(a,1),4);
        linePos(1:size(a,1),1)=65; 
        linePos(1:size(a,1),3)=207;  
    end
    
    
   for t=1:size(linePos,1)
       if(a(t)>0 & abs(a(t))~=Inf)
        linePos(t,2)=round(linePos(t,1)*a(t)+b(t));
        linePos(t,4)=round(linePos(t,3)*a(t)+b(t));
        if(linePos(t,2)>=size(im,1) || linePos(t,4)>=size(im,2))
            delete=delete+1;
        end
       else
           delete=delete+1;
       end
   end
   
   
   if(delete>0)
      linePos2=zeros(size(linePos,1)-delete,4); 
      a2=zeros(size(linePos,1)-delete,1);
      b2=zeros(size(linePos,1)-delete,1);
      k=1;
       for t=1:size(linePos,1)
          
            if(linePos(t,2)<size(im,1) && linePos(t,4)<size(im,2)&& a(t)>0)
                linePos2(k,:)=linePos(t,:);
                a2(k)=a(t);
                b2(k)=b(t);
                k=k+1;
            end
       end
       
       
      linePos=linePos2;
      a=a2;
      b=b2;
   end
   
   if(size(a,1)==0 )
        b=[295.6260,230.52,173.9412,206.8865,134.4157]';
        a=[0.0153,-0.0200,0.0471,0.0922,0.3072]';
        linePos=zeros(size(a,1),4);
        linePos(1:size(a,1),1)=65; 
        linePos(1:size(a,1),3)=207;  
        linePos(1:size(a,1),2)=round(linePos(1:size(a,1),1).*a(1:size(a,1))+b(1:size(a,1)));
        linePos(1:size(a,1),4)=round(linePos(1:size(a,1),3).*a(1:size(a,1))+b(1:size(a,1)));
        
        
    end
   
   
   a(1:end)=mean(a);

   [sorted_b, Indexes]=sort(b);
   
   linePos2=zeros(size(linePos,1),4);
   
   for n=1:size(linePos,1)
       linePos2(n,:)=linePos(Indexes(n),:);     
   end
   
    
    lineFrame = insertShape(im,'Line',linePos2,...
                'Color','yellow','LineWidth',5);

    figure
    imshowpair(im, lineFrame, 'montage'); 
  
    b=sorted_b;
   linePos=linePos2;
   
   
    endColumn=linePos(:,3);
    maxend=min(endColumn);
    linePos(:,3)=maxend;
%     
    startColumn=linePos(:,1);
    maxstart=min(startColumn);
    linePos(:,1)=maxstart;

    delete=0;
    
    for t=1:size(linePos,1)
        linePos(t,2)=round(linePos(t,1)*a(t)+b(t));
        linePos(t,4)=round(linePos(t,3)*a(t)+b(t));
        
        if(linePos(t,2)>size(im,1) || linePos(t,4)>size(im,2))
            delete=delete+1;
        end
        
    end
 
    if(delete>0)
      linePos2=zeros(size(linePos,1)-delete,4); 
      a2=zeros(size(linePos,1)-delete,1);
      b2=zeros(size(linePos,1)-delete,1);
      k=1;
       for t=1:size(linePos,1)
          
            if(linePos(t,2)<size(im,1) && linePos(t,4)<size(im,2))
                linePos2(k,:)=linePos(t,:);
                a2(k)=a(t);
                b2(k)=b(t);
                k=k+1;
            end
       end
      linePos=linePos2;
      a=a2;
      b=b2;
   end
    
    
    
    
    lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',5);

    figure
    imshowpair(im, lineFrame, 'montage'); 


max1=0;
min1=300;
idx=-1;

max2=0;
min2=300;
idx2=-1;


L1=15;
row_start=-1;
t=1;
while(t<round(size(BW2,1)/2) && row_start<0 )
    for s=1:size(BW2,1)-L1
        p=BW2(t:t+L1,s:s+L1);
        pp=diag(p);
        if(pp(:,1)==1)
            row_start=t;
%             pp=temp2(t,s:s+L1);
            continue;
        end
    end
    t=t+1;
end


if(max(linePos(:,2))<266 && max(linePos(:,4))<266)
    
    newLinePos=zeros(size(linePos,1)+1,4);
    newLinePos(1:end-1,:)=linePos;
    
    newb=zeros(size(b,1)+1,1);
    newb(1:end-1)=b;
    
    newb(end)=newb(end-1)+min(b);
    newa=zeros(size(a,1)+1,1);
    newa(1:end-1)=a;
    newa(end)=newa(end-1);
    
    
    
    newLinePos(end,:)=linePos(end,:);
    newLinePos(end,2)=round(linePos(end,1)*newa(end)+newb(end));
    newLinePos(end,4)=round(linePos(end,3)*newa(end)+newb(end));
    linePos=newLinePos;
    b=newb;
    a=newa;
end


%%%sort from here 

if(min(b)>25)
    
    newLinePos=zeros(size(linePos,1)+1,4);
    newLinePos(2:end,:)=linePos;
    newb=zeros(size(b,1)+1,1);
    newb(2:end)=b;
    newLinePos(1,:)=linePos(1,:);
    %%%%%%%%%%%%%%
    newLinePos(1,2)=45;
    %%%%%%%%%%%%%%%%%%%
    x=newLinePos(1,1);
    y=newLinePos(1,2);
    newb(1)=round(y-a(1)*x);
    newa=zeros(size(a,1)+1,1);
    newa(2:end)=a;
    newa(1)=newa(end-1);
    newLinePos(1,4)=round(linePos(1,3)*newa(1,:)+newb(1));
    linePos=newLinePos;
    b=newb;
    a=newa;
end



   lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',5);

    figure
    imshowpair(im, lineFrame, 'montage'); 



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



   lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',5);

    figure
    imshowpair(im, lineFrame, 'montage'); 


go=1;
min_final=min1;

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
         
        if((abs(min_final-min1)<5)&& addedline>1.5)
%             while(p<addedline)
                for y=1:idx
                    newLinePos(y,:)=linePos(y,:);
                end
                
                newa=zeros(size(a,1)+1,1);
                newa(2:end)=a;
                newa(1)=newa(end-1);

                newb=zeros(size(b,1)+1,1);
                newb(1:idx)=b(1:idx);
                newb(idx+1)=newb(idx)+min1;
                newb(idx+2:end)=b(idx+1:end);
                newLinePos(idx+1,1)=linePos(idx,1);
                newLinePos(idx+1,3)=linePos(idx,3);

                x1=round(linePos(idx+1,1)*newa(1,:)+newb(idx+1));
                x3=round(linePos(idx+1,3)*newa(1,:)+newb(idx+1));

                newLinePos(idx+1,2)=x1;
                newLinePos(idx+1,4)=x3;
             
                newLinePos(idx+2:end,:)=linePos(idx+1:end,:);
                linePos=newLinePos;
                b=newb;
                a=newa;
 
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


   lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',5);

    figure
    imshowpair(im, lineFrame, 'montage'); 


   lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',5);

    close all;
    figure
    imshow(lineFrame); 
  

end

end
% 
% fclose(fileID);
% fclose(fileID2);
