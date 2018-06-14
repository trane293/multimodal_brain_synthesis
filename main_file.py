from loader_multimodal import Data
from runner import Experiment

data = Data('./BRATS', dataset='BRATS', trim_and_downsample=False, modalities_to_load=['T1', 'T2', 'T2FLAIR'])
data.load()

input_modalities = ['T1', 'T2', 'T2FLAIR']
output_weights = {'T2FLAIR': 1.0, 'concat': 1.0}
exp = Experiment(input_modalities, output_weights, './RESULTS', data, latent_dim=16, spatial_transformer=False)
exp.run(data)