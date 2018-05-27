import h5py
import numpy as np
import pandas as pd

#d = np.loadtxt('keypoint_train.txt')
motgt = pd.read_csv("keypoint_test.txt", sep=" ", index_col = False)
#motgt = motgt.loc[:,'1':'41']
motgt = motgt.drop(motgt.columns[[0,41]],axis=1)
final = motgt.as_matrix()
#np.savetxt('part_valid.txt', final, delimiter=',',fmt='%.0f')
print(np.shape(final))
final=np.reshape(final,(241560,2))
#final=final[0:2958]
np.savetxt('part_valid.txt', final, delimiter=',',fmt='%.0f')
DATASET1='istrain'
FILE1='../mpii_original/annot/valid.h5'
#file1 = h5py.h5f.open(FILE1)
#dset1 = h5py.h5d.open(file1, DATASET1)

#print(dset1)
h = h5py.File('data2_test.h5', 'w')
dset = h.create_dataset('deprecated', data=final)
#dset22 = h.create_dataset('istrain', data=dset1)
fs = h5py.File('../mpii_original/annot/valid.h5', 'r')
fs2 = h5py.File('../mpii_original/annot.h5', 'r')
fs3=h5py.File('person_valid.h5', 'r')
fs4=h5py.File('imgname_test6.h5', 'r')
fs5=h5py.File('part_valid.h5', 'r')
fs6 = h5py.File('center_valid.h5', 'r')
#fs8 = h5py.File('index_subset.h5', 'r')
fs7=h5py.File('partvalid16.h5', 'r')
#fd = h5py.File('data2.h5', 'w')
#h.create_group('istrain')
#fs2.copy('istrain', h)
fs.copy('index', h)
fs4.copy('imgname', h)
fs.copy('scale', h)
fs3.copy('person',h)
#fs5.copy('part',h)
fs6.copy('center',h)
fs7.copy('part',h)
