import h5py
import numpy as np

'''
if 't1.' 
    i = 0
    seq_name = 't1'
elif 't2.' in imagefile:
    i = 1
    seq_name = 't2'
elif 't1ce.' in imagefile:
    i = 2
    seq_name = 't1ce'
elif 'flair.' in imagefile:
    i = 3
    seq_name = 'flair'
'''

mount_path_prefix = '/home/anmol/'
hdf5_filepath = mount_path_prefix + 'BRATS_Combined.h5'
hdf5_file = h5py.File(hdf5_filepath, 'r')
hf = hdf5_file['combined']
hgg_data = hf['training_data']

t1 = hgg_data[:,0,...]
t1 = np.swapaxes(t1, 3, 2)
t1 = np.swapaxes(t1, 2, 1)
np.save(open('T1.npz', 'wb'), t1)
del t1

t2 = hgg_data[:,1,...]
t2 = np.swapaxes(t2, 3, 2)
t2 = np.swapaxes(t2, 2, 1)
np.save(open('T2.npz', 'wb'), t2)
del t2

t1ce = hgg_data[:,2,...]
t1ce = np.swapaxes(t1ce, 3, 2)
t1ce = np.swapaxes(t1ce, 2, 1)
np.save(open('T1CE.npz', 'wb'), t1ce)
del t1ce

t2flair = hgg_data[:,3,...]
t2flair = np.swapaxes(t2flair, 3, 2)
t2flair = np.swapaxes(t2flair, 2, 1)
np.save(open('T2FLAIR.npz', 'wb'), t2flair)
del t2flair

print('hello')

