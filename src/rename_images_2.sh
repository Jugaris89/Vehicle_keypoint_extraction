#!/bin/bash
FILES=/mnt/md0/jgarcia/pose-hg-train/data/mpii/images
FILES2=/mnt/md0/jgarcia/pose-hg-train/data/mpii_original/images
FILE_O=/mnt/md0/jgarcia/pose-hg-train/data/mpii/images/

for entry in $FILES/*
do
  echo "$entry"|grep "_c" >> images.txt
done

for entry in $FILES2/*
do
  echo "$entry" >> images_original.txt
done


