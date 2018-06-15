from loader_multimodal import Data
from runner import Experiment
import optparse

parser = optparse.OptionParser()
parser.add_option('--dir', '--directory',
                  dest="resultsdir",
                  default='./RESULTS',
                  type='str',
                  help='Directory to save the results'
                  )

parser.add_option('--exp', '--experiment',
                  dest="experiment",
                  default=0,
                  type='int',
                  help='Which experiment to perform'
                  )


options, remainder = parser.parse_args()

if options.experiment == 0:
	print('Training model with 2 inputs and 1 output')
	data = Data('./BRATS', dataset='BRATS', trim_and_downsample=False, modalities_to_load=['T1', 'T2', 'T2FLAIR'])
	data.load()

	input_modalities = ['T1', 'T2']
	output_weights = {'T2FLAIR': 1.0, 'concat': 1.0}
	exp = Experiment(input_modalities, output_weights, options.resultsdir, data, latent_dim=16, spatial_transformer=True)
	exp.run(data)
else:
	print('Training model with 4 inputs and 4 outputs')
	data = Data('./BRATS', dataset='BRATS', trim_and_downsample=False, modalities_to_load=['T1', 'T2', 'T1CE', 'T2FLAIR'])
        data.load()

        input_modalities = ['T1', 'T2', 'T1CE', 'T2FLAIR']
        output_weights = {'T1': 1.0, 'T2': 1.0, 'T1CE': 1.0, 'T2FLAIR': 1.0, 'concat': 1.0}
        exp = Experiment(input_modalities, output_weights, options.resultsdir, data, latent_dim=16, spatial_transformer=True)
        exp.run(data)
