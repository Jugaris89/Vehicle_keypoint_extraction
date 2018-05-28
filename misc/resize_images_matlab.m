% for i=1:38551
% l=convertStringsToChars(keypointtrainid(i));
% s=strcat('/mnt/md0/jgarcia/pose-hg-train/data/mpii/images/',l);
% s1=strcat('/mnt/md0/jgarcia/pose-hg-train/data/mpii/images_256/',l);
% j=imread(s);
% j1=imresize(j,[256,256]);
% imwrite(j1,s1);
% end
% for i=1:12077
% l=convertStringsToChars(keypointtestid(i));
% s=strcat('/mnt/md0/jgarcia/pose-hg-train/data/mpii/images/',l);
% s1=strcat('/mnt/md0/jgarcia/pose-hg-train/data/mpii/images_256/',l);
% j=imread(s);
% j1=imresize(j,[256,256]);
% imwrite(j1,s1);
% end

for i=1:1678
l=convertStringsToChars(namequery(i));
s=strcat('/mnt/md0/jgarcia/pose-hg-train/data/mpii/images/',l);
s1=strcat('/mnt/md0/jgarcia/pose-hg-train/data/mpii/images_256/',l);
j=imread(s);
j1=imresize(j,[256,256]);
imwrite(j1,s1);
end
