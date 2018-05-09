#!/bin/bash
FILES=/mnt/md0/jgarcia/pose-hg-train/data/mpii/images/*
FILES2=/mnt/md0/jgarcia/pose-hg-train/data/mpii_original/images/*
FILE_O=/mnt/md0/jgarcia/pose-hg-train/data/mpii/images
for f in $(cat images.txt)
do
#  echo $f
  for m in $(cat images_original.txt)
  do
      j=`echo "${m##*/}"`
      F=`echo "${j%%.*}"`
      echo $f 
      echo $FILE_O'/'$F'.jpg'
      cp $f $FILE_O'/'$F'.jpg'
      sed '1d' images_original.txt > tmpfile; mv tmpfile images_original.txt
      break
  done
done
