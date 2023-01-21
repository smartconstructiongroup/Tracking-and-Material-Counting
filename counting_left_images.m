%This section is related to counting images
%This code is related to Left View Images

clc;
clear all;
close all;

Directory = '0\';
% Read images from Images folder
Imgs = dir(fullfile(Directory,'*.jpg'));

% fileID = fopen('0\results.txt','w');
% fileID = fopen('1\loutput.txt','w');
% fileID2 = fopen('1\lgroundtruth.txt','w');
for j=1:length(Imgs)
im = imread(fullfile(Directory,Imgs(j).name)); % Read image

label=0;

if(label==0)
    I=im;
    
    T =edge(I,'canny',0.1);
    BW = T;
    
    figure;
    imshowpair(I, BW, 'montage')
    
    %% Detect Lines
    % Perform Hough Transform
    [H,T,R] = hough(BW);

    
    % Identify Peaks in Hough Transform
    hPeaks =  houghpeaks(H,20,'NHoodSize',[19 19]);

    % Extract lines from hough transform and peaks
    hLines = houghlines(BW,T,R,hPeaks,...
            'FillGap',100,'MinLength',50);

    %% View results
    % Overlay lines
    [linePos,markerPos] = getVizPosArray(hLines);


    lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',4);

    figure
    imshowpair(im, lineFrame, 'montage'); 
    
    
    
    a=zeros(size(linePos,1),1);
    b=zeros(size(linePos,1),1);
    
    for t=1:size(linePos,1)
        coefficients = polyfit([linePos(t,1), linePos(t,3)], [linePos(t,2), linePos(t,4)], 1);
        a(t) = coefficients (1);
        b(t) = coefficients (2);
    end
  
    
    number_noise=-1;
    threshold=0.04;
    t=1;
    go=1;
    
    linePos2=zeros(size(linePos,1)-1,4);
    c=1;
    cc=1;
    a2=zeros(size(a,1),1);
    b2=zeros(size(b,1),1);
    
    
    
    
    while(go==1)
        
        if(a(t,1)<0 & abs(a(t,1))~=Inf)

            linePos2(cc,:)=linePos(c,:);
            a2(cc)=a(c);
            b2(cc)=b(c);
            cc=cc+1;

        end
         t=t+1;
         c=c+1;
        if(t>size(a,1))
            go=0;
        end       
    end
    
    a=a2(1:cc-1,:);
    b=b2(1:cc-1,:); 
    
    if(cc-1==2)
        [sorted_a, Indexes]=sort(a);
        linePos=linePos2(Indexes(1),:);
        a=a2(Indexes(1),:);
        b=b2(Indexes(1),:);
    elseif(cc-1==0)
    
        newLinePos=zeros(1,4);

        newLinePos(1,1)=170;
        newLinePos(1,2)=298;
        newLinePos(1,3)=252;
        newLinePos(1,4)=265;
        linePos=newLinePos;
        
        a=zeros(size(linePos,1),1);
        b=zeros(size(linePos,1),1);

        for t=1:size(linePos,1)
            coefficients = polyfit([linePos(t,1), linePos(t,3)], [linePos(t,2), linePos(t,4)], 1);
            a(t) = coefficients (1);
            b(t) = coefficients (2);
        end
        
        
    else
    
        linePos=linePos2(1:cc-1,:);

    end
   
    
    a_p=median(a(:));
    
    if(a_p>-0.09 && a_p<0)
        a_p=-0.4;
        
    end
    
    
    delete=0;
   for t=1:size(linePos,1)
        a(t) = a_p;
        linePos(t,2)=round(linePos(t,1)*a(t)+b(t));
        linePos(t,4)=round(linePos(t,3)*a(t)+b(t));
        
        if(linePos(t,2)>size(im,1) || linePos(t,4)>size(im,2) ||linePos(t,2)<=0 || linePos(t,4)<=0 )
            delete=delete+1;
        end
        
        
   end
   
   
  if(delete>0)
      linePos2=zeros(size(linePos,1)-delete,4); 
      a2=zeros(size(linePos,1)-delete,1);
      b2=zeros(size(linePos,1)-delete,1);
      k=1;
       for t=1:size(linePos,1)
          
            if(linePos(t,2)<size(im,1) && linePos(t,4)<size(im,2)&& a(t)<0 && linePos(t,2)>0 && linePos(t,4)>0)
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
        b=333.12;
        a=-1.43;
        linePos=zeros(1,4);
        linePos(1:size(a,1),1)=191; 
        linePos(1:size(a,1),3)=59;  
        linePos(1:size(a,1),2)=round(linePos(1:size(a,1),1).*a(1:size(a,1))+b(1:size(a,1)));
        linePos(1:size(a,1),4)=round(linePos(1:size(a,1),3).*a(1:size(a,1))+b(1:size(a,1)));
        
        
    end
   
   
   a(1:end)=median(a);
  
   
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
   

    linePos(:,3)=170;
    
    linePos(:,1)=130;

    for t=1:size(linePos,1)
        linePos(t,2)=round(linePos(t,1)*a(t)+b(t));
        linePos(t,4)=round(linePos(t,3)*a(t)+b(t));
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



if(max(linePos(:,2))<265 && max(linePos(:,4))<265)
    
    newLinePos=zeros(size(linePos,1)+1,4);
    newLinePos(1:end-1,:)=linePos;
    
    newb=zeros(size(b,1)+1,1);
    newb(1:end-1)=b;
    
    newb(end)=newb(end-1)+min(b);
    if(newb(end)>500)
        newb(end)=460;
    end
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

    lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',5);

    figure
    imshowpair(im, lineFrame, 'montage'); 

if(min(b)>310)
    
    newLinePos=zeros(size(linePos,1)+1,4);
    newLinePos(2:end,:)=linePos;
    
    newb=zeros(size(b,1)+1,1);
    newb(2:end)=b; 

    newb(1)=90;
    
    newa=zeros(size(a,1)+1,1);
    newa(2:end)=a;
    newa(1)=newa(end);
    
    
    newLinePos(1,:)=linePos(1,:);
    
    newLinePos(1,2)=round(linePos(1,1)*newa(1,:)+newb(1));
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
        idx_min=k;
    end

    if(temp2>max2)
        max2=temp2;
        idx2=k;
    end
    if(temp2<min2)
        min2=temp2;
        idx2_min=k;
    end


end


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
        if(size(linePos,1)==2 && min1>100)
            addedline=200;
%             min1=60;
        else
            addedline=round(max1/min1);
        end
        newLinePos=zeros(size(linePos,1)+1,4);
%         p=0;
        
        
        if((abs(min_final-min1)<3)&& addedline>1)
%             while(p<addedline)
                for y=1:idx
                    newLinePos(y,:)=linePos(y,:);
                end
                
                if(addedline==200)
                    min1=24;
                    min2=24;
                    min_final=min1;
                end

                newLinePos(idx+1,2)=linePos(idx,2)+min1;
                newLinePos(idx+1,4)=linePos(idx,4)+min2;
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



   lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',5);

    figure
    imshowpair(im, lineFrame, 'montage'); 


go=1;

while(go)
   if (size(linePos,1)<15)
    max1=0;
    min1=300;
    idx=-1;

    max2=0;
    min2=300;
    idx2=-1;

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

    if(min1<10 && min2<10)
        newLinePos=zeros(size(linePos,1)-1,4);
        newLinePos(1:idx-1,:)=linePos(1:idx-1,:);
        newLinePos(idx:end,:)=linePos(idx+1:end,:);
        linePos=newLinePos;
    else
        go=0;
    end
   else
       break;
   end
end
  close all;
%%%%%%%%%%
    lineFrame = insertShape(im,'Line',linePos,...
                'Color','yellow','LineWidth',5);

    figure
    imshow(lineFrame);
  
end

end

% fclose(fileID);
% fclose(fileID2);