
import os
import math
import numpy as np
import csv
import ast
from skimage.measure import block_reduce
from scipy.ndimage.interpolation import rotate, shift
import nibabel as nib
import random
from random import shuffle
random.seed(1338)

class Data(object):
    '''
    Class used to load data for ISLES, BRATS and IXI datasets. ISLES2015 and BRATS data can be downloaded from
    http://www.isles-challenge.org/ISLES2015 and https://sites.google.com/site/braintumorsegmentation/home/brats2015.
    This loader expects the data to be in a .npz format where there's one .npz file for each modality, e.g. T1.npz, T2.npz etc.
    Each .npz contains an array of volumes, and every volume has shape (Z, H, W), where Z is the number of spatial slices 
    and H, W the height and width of a slice respectively.
    IXI dataset can be downloaded from http://brain-development.org/ixi-dataset and can be loaded as-is.

    A splits.txt file is expected if the Data object will be used for cross-validation through runner.py.
    An example splits.txt could have the following contents:
    test,validation,train
    "[0,1]","[2,3]","[4,5]"

    Example usage of the class:
    data = Data('./data/ISLES', modalities_to_load=['T1','T2'], dataset='ISLES', trim_and_downsample=False)
    data.load()
    vols_0_1_2 = data.select_for_ids('T1', [0, 1, 2])
    '''
    def __init__(self, data_folder, modalities_to_load=None, dataset='ISLES', trim_and_downsample=False, normalize_volumes=True):
        self.data_folder = data_folder[:-1] if data_folder.endswith('/') else data_folder

        self.dataset = dataset
        self.normalize_volumes = normalize_volumes
        if self.dataset == 'ISLES':
            self.num_vols = 28
            self.splits_file = './splits.txt'
        elif self.dataset == 'BRATS':
            self.num_vols = 54
            self.splits_file = './splits_lgg.txt'
        elif self.dataset == 'IXI':
            self.num_vols = 28
            self.splits_file = './splits.txt'

        if modalities_to_load is not None:
            self.modalities_to_load = modalities_to_load
        elif dataset == 'ISLES':
            self.modalities_to_load = ['T1', 'T2', 'T2FLAIR', 'DWI', 'MASK']
        elif dataset == 'BRATS':
            self.modalities_to_load = ['T1', 'T2', 'T2FLAIR']
        elif dataset == 'IXI':
            self.modalities_to_load = ['T2', 'PD']

        self.T1 = None
        self.T2 = None
        self.T2FLAIR = None
        self.T1CE = None
        self.DWI = None
        self.MASK = None

        self.channels = dict()
        self.rotations = {mod: False for mod in self.modalities_to_load}
        self.shifts = {mod: False for mod in self.modalities_to_load}
        self.refDict = {'T1': self.T1, 'T2': self.T2, 'T2FLAIR': self.T2FLAIR, 'T1CE': self.T1CE,
                        'DWI': self.DWI, 'MASK': self.MASK}
        self.trim_and_downsample = trim_and_downsample

    def load(self):
        for mod_name in self.modalities_to_load:
            print 'Loading ' + mod_name
            norm_vols = False if mod_name == 'MASK' else self.normalize_volumes
            mod = self.load_modality(mod_name, normalize_volumes=norm_vols,
                                     rotate_mult=self.rotations[mod_name],
                                     shift_mult=self.shifts[mod_name])

            self.refDict[mod_name] = mod

        if 'T1' in self.refDict:
            self.T1 = self.refDict['T1']

        if 'T2' in self.refDict:
            self.T2 = self.refDict['T2']

        if 'T2FLAIR' in self.refDict:
            self.T2FLAIR = self.refDict['T2FLAIR']

        if 'T1CE' in self.refDict:
            self.T1CE = self.refDict['T1CE']

        if 'DWI' in self.refDict:
            self.DWI = self.refDict['DWI']

        if 'MASK' in self.refDict:
            self.MASK = self.refDict['MASK']

    def set_rotate(self, modality, mult=1.0):
        self.rotations[modality] = mult

    def set_shift(self, modality, mult=1.0):
        self.shifts[modality] = mult

    def remove_volume(self, vol):
        if self.T1 is not None:
            del self.T1[vol]
        if self.T2 is not None:
            del self.T2[vol]
        if self.T2FLAIR is not None:
            del self.T2FLAIR[vol]
        if self.DWI is not None:
            del self.DWI[vol]
        if self.MASK is not None:
            del self.MASK[vol]

        self.num_vols -= 1

    def load_ixi(self, mod):
        folder = self.data_folder + '/IXI-' + mod
        data = [nib.load(folder + '/' + f).get_data() for f in np.sort(os.listdir(folder))]
        data = [np.swapaxes(np.swapaxes(d, 1, 2), 0, 1) for d in data]
        print 'Loaded %d vols from IXI' % len(data)
        return data

    def load_modality(self, modality, normalize_volumes=True, downsample=2, rotate_mult=0.0, shift_mult=0.0):

        if self.dataset == 'ISLES':
            file_name = self.data_folder + '/ISLES/' + modality + '.npz'
            data = np.load(file_name)['arr_0']
        elif self.dataset == 'BRATS':
            file_name = self.data_folder + '/' + modality + '.npz'
            data = np.load(file_name)
        elif self.dataset == 'IXI':
            data = self.load_ixi(modality)
        else:
            raise Exception('Unknown dataset ', self.dataset)

        # array of 3D volumes
        X = [data[i].astype('float32') for i in range(self.num_vols)]

        # trim the matrices and downsample: downsample x downsample -> 1x1
        for i, x in enumerate(X):

            if rotate_mult != 0:
                print 'Rotating ' + modality + '. Multiplying by ' + str(rotate_mult)
                rotations = [[-5.57, 2.79, -11.99], [-5.42, -18.34, -14.22], [4.64, 5.80, -5.96],
                             [-17.02, -8.70, 15.43],
                             [18.79, 17.44, 17.06], [-14.55, -4.90, 9.19], [14.37, -0.58, -16.85],
                             [-9.49, -12.53, -2.89],
                             [-16.75, -4.07, 3.23], [14.39, -16.58, 3.35], [-14.05, -2.25, -10.58],
                             [8.47, -8.95, -12.73],
                             [13.00, -10.90, -2.85], [2.61, -7.51, -6.26], [-13.99, -0.38, 6.29],
                             [10.16, -9.88, -11.89],
                             [6.76, 0.83, -19.85], [18.74, -6.70, 15.46], [-3.01, -2.85, 18.45], [-17.37, -1.32, -3.48],
                             [14.67, -17.93, 18.74], [6.55, 18.19, -8.24], [13.52, -4.09, 19.32], [5.27, 11.27, 4.93],
                             [2.29, 17.83, 10.07], [-11.98, 10.49, 0.02], [14.49, -12.00, -17.21],
                             [17.86, -17.38, 19.04]]
                theta = rotations[i]

                x = rotate(x, rotate_mult * theta[0], axes=(1, 0), reshape=False, order=3, mode='constant', cval=0.0,
                           prefilter=True)
                x = rotate(x, rotate_mult * theta[1], axes=(1, 2), reshape=False, order=3, mode='constant', cval=0.0,
                           prefilter=True)
                x = rotate(x, rotate_mult * theta[2], axes=(0, 2), reshape=False, order=3, mode='constant', cval=0.0,
                           prefilter=True)

            if shift_mult != 0:
                print 'Shifting ' + modality + '. Multiplying by ' + str(shift_mult)
                shfts = [[0.931, 0.719, -0.078], [0.182, -0.220, 0.814], [0.709, 0.085, -0.262], [-0.898, 0.367, 0.395],
                         [-0.936, 0.591, -0.101], [0.750, 0.522, 0.132], [-0.093, 0.188, 0.898],
                         [-0.517, 0.905, -0.389],
                         [0.616, 0.599, 0.098], [-0.209, -0.215, 0.285], [0.653, -0.398, -0.153],
                         [0.428, -0.682, -0.501],
                         [-0.421, -0.929, -0.925], [-0.753, -0.492, 0.744], [0.532, -0.302, 0.353],
                         [0.139, 0.991, -0.086],
                         [-0.453, 0.657, 0.072], [0.576, 0.918, 0.242], [0.889, -0.543, 0.738], [-0.307, -0.945, 0.093],
                         [0.698, -0.443, 0.037], [-0.209, 0.882, 0.014], [0.487, -0.588, 0.312],
                         [0.007, -0.789, -0.107],
                         [0.215, 0.104, 0.482], [-0.374, 0.560, -0.187], [-0.227, 0.030, -0.921], [0.106, 0.975, 0.997]]
                shft = shfts[i]
                x = shift(x, [shft[0] * shift_mult, shft[1] * shift_mult, shft[2] * shift_mult])

            if self.dataset == 'ISLES':
                x = x[:, 0:-6, 34:-36]

            if self.trim_and_downsample:
                X[i] = block_reduce(x, block_size=(1, downsample, downsample), func=np.mean)

                if self.dataset == 'BRATS':
                    # power of 2 padding
                    (_, w, h) = X[i].shape

                    w_pad_size = int(math.ceil((math.pow(2, math.ceil(math.log(w, 2))) - w) / 2))
                    h_pad_size = int(math.ceil((math.pow(2, math.ceil(math.log(h, 2))) - h) / 2))

                    X[i] = np.lib.pad(X[i], ((0, 0), (w_pad_size, w_pad_size), (h_pad_size, h_pad_size)), 'constant',
                                      constant_values=0)

                    (_, w, h) = X[i].shape

                    # check if dimensions are even

                    if w & 1:
                        X[i] = X[i][:, 1:, :]

                    if h & 1:
                        X[i] = X[i][:, :, 1:]

            else:
                X[i] = x

        if normalize_volumes:
            for i, x in enumerate(X):
                X[i] = X[i] / np.mean(x)

        if rotate_mult > 0:
            for i, x in enumerate(X):
                X[i][X[i] < 0.25] = 0

        return X

    def add_channel(self, modality, channel):
        assert modality in self.refDict
        assert channel in self.refDict
        self.channels.update({modality: channel})

    def select_for_ids(self, modality, ids, as_array=True):
        assert modality in self.refDict

        data_ids = [self.refDict[modality][i] for i in ids]

        if as_array:
            data_ids_ar = np.concatenate(data_ids)
            if len(data_ids_ar.shape) < 4:
                data_ids_ar = np.expand_dims(data_ids_ar, axis=1)
            if modality in self.channels:
                ch_ids = [self.refDict[self.channels[modality]][i] for i in ids]
                ch_ids_ar = np.expand_dims(np.concatenate(ch_ids), axis=1)
                return np.concatenate([data_ids_ar, ch_ids_ar], axis=1)
            else:
                return data_ids_ar
        else:
            data_ids_ar = data_ids
            if len(data_ids_ar[0].shape) < 4:
                data_ids_ar = [np.expand_dims(d, axis=1) for d in data_ids]
            if modality in self.channels:
                ch_ids = [self.refDict[self.channels[modality]][i] for i in ids]
                ch_ids_ar = [np.expand_dims(ch, axis=1) for ch in ch_ids]
                return [np.concatenate([data_ids_ar[i], ch_ids_ar[i]], axis=1) for i in range(len(ids))]
            else:
                return data_ids_ar

    def id_splits_iterator_original(self):
        # return a dictionary of train, validation and test ids
        with open(self.splits_file, 'r') as f:
            r = csv.reader(f, delimiter=',', quotechar='"')
            headers = next(r)
            for row in r:
                if len(row) == 0:
                    break
                if row[0].startswith('#'):
                    continue

                yield {headers[i].strip(): ast.literal_eval(row[i]) for i in range(len(headers))}


    def id_splits_iterator(self):
        # # generate the index lsit identical to the BRATS framework. Hardcoded most values in orde rto avoid interfacing
        # # a non-trivial amount of code.
        # # 285 = number of patients
        # # 97 = Percentage of training examples
        # # add some validation patients from the test set.
        # indices = list(range(0, 75))
        # shuffle(indices)
        # train_end = int((len(indices) * 97) / 100.0)
        # train_indices = indices[0:train_end]
        # val_indices = indices[train_end:train_end + 3]
        # test_indices = indices[train_end + 3:]
        # return {'train': train_indices, 'validation': val_indices, 'test': test_indices}
        return {'train': list(range(0, 50)), 'validation': list(range(50, 52)), 'test': list(range(52, 54))}
