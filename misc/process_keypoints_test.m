for i=1:12077
l=convertStringsToChars(keypointtestid(i));
s=strcat('/mnt/md0/jgarcia/pose-hg-train/data/mpii/images_good/',l);
j=imread(s);
[x1,y1,z1] = size(j);
if (y1 > 256)
    for j=1:40
        if ((rem(j,2)==1) && (keypointnotationtest(i,j)>0))
            keypointnotationtest(i,j)=round((256/y1)*keypointnotationtest(i,j));
        end
    end
    if (x1 > 256)
        for j=1:40
            if ((rem(j,2)==0) && (keypointnotationtest(i,j)>0))
                keypointnotationtest(i,j)=round((256/x1)*keypointnotationtest(i,j));
            end
        end
    end
end

if (y1 < 256)
    for j=1:40
        if ((rem(j,2)==1) && (keypointnotationtest(i,j)>0))
            keypointnotationtest(i,j)=round((256/y1)*keypointnotationtest(i,j));
        end
    end
    if (x1 < 256)
        for j=1:40
            if ((rem(j,2)==0) && (keypointnotationtest(i,j)>0))
                keypointnotationtest(i,j)=round((256/x1)*keypointnotationtest(i,j));
            end
        end
    end
end
end