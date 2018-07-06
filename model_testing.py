from loader_multimodal import Data
from runner import Experiment
import optparse
import platform
import os
import matplotlib.pyplot as plt
import numpy as np

def viewCurrentVolume(preds):
    fig, ax = plt.subplots(2, 5)
    ax = list(ax[0]) + list(ax[-1])
    for idx, i in enumerate(range(0, 150, 15)):
        ax[idx].imshow(preds[i,0,:,:], cmap='gray')
    plt.show()

# to make the code portable even on cedar,you need to add conditions here
node_name = platform.node()
if node_name == 'XPS15':
    # this is my laptop, so the cedar-rm directory is at a different place
    # mount_path_prefix = '/home/anmol/mounts/cedar-rm/'
    data_dir = './npz_BRATS'
    model_path = '/home/anmol/exp_files/'
    # model_path = './'

elif 'computecanada' in node_name: # we're in compute canada, maybe in an interactive node, or a scheduler node.
    mount_path_prefix = '/home/asa224/' # home directory
    data_dir = os.path.join(mount_path_prefix, 'scratch/asa224/Datasets/npz_BRATS')

print('Testing the model')
data = Data(data_dir, dataset='BRATS', trim_and_downsample=False, modalities_to_load=['T1', 'T2', 'T2FLAIR'], normalize_volumes=False)
data.load()

input_modalities = ['T1', 'T2']
output_weights = {'T2FLAIR': 1.0, 'concat': 1.0}
exp = Experiment(input_modalities, output_weights, './RESULTS', data, latent_dim=16, spatial_transformer=True)

load_input_modalities = ['T1', 'T2']
load_output_modalities = 'T2FLAIR'

exp.load_partial_model(folder=model_path, model_name='model', input_modalities=load_input_modalities,
                       output_modality=load_output_modalities)
predictions = exp.run_test_minimal(data)

string = '_'.join(load_input_modalities) + '-->' + load_output_modalities

for i in range(len(predictions)):
    np.save(open('./PREDICTIONS/pred_{}_[{}].npz'.format(string, i), 'wb'), predictions[i])

print('Hello')
