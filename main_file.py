from loader_multimodal import Data
from runner import Experiment
import optparse
import platform
import os

# to make the code portable even on cedar,you need to add conditions here
node_name = platform.node()
if node_name == 'XPS15':
    # this is my laptop, so the cedar-rm directory is at a different place
    # mount_path_prefix = '/home/anmol/mounts/cedar-rm/'
    data_dir = './npz_BRATS'
elif 'computecanada' in node_name: # we're in compute canada, maybe in an interactive node, or a scheduler node.
    mount_path_prefix = '/home/asa224/' # home directory
    data_dir = os.path.join(mount_path_prefix, 'scratch/asa224/Datasets/npz_BRATS')

parser = optparse.OptionParser()
parser.add_option('--dir', '--directory',
                  dest="resultsdir",
                  default='./RESULTS',
                  type='str',
                  help='Directory to save the results'
                  )

parser.add_option('--exp', '--experiment',
                  dest="experiment",
                  default=1,
                  type='int',
                  help='Which experiment to perform'
                  )

parser.add_option('--c', '--checkpoint',
                  dest="checkpoint",
                  default=None,
                  type='str',
                  help='Path to the latest checkpoint file to resume training from'
                  )


options, remainder = parser.parse_args()

# options.checkpoint = os.path.join(mount_path_prefix, 'rrg_proj_dir/multimodal_brain_synthesis/RESULTS/split0/model_0')
if options.experiment == 0:
    print('Training model with 2 inputs and 1 output')
    data = Data(data_dir, dataset='BRATS', trim_and_downsample=False, modalities_to_load=['T1', 'T2', 'T2FLAIR'], normalize_volumes=False)
    data.load()
    input_modalities = ['T1', 'T2']
    output_weights = {'T2FLAIR': 1.0, 'concat': 1.0}
    exp = Experiment(input_modalities, output_weights, options.resultsdir, data, latent_dim=16, spatial_transformer=True)

    if options.checkpoint != None:
        exp.resume_from_checkpoint(data, options.checkpoint)
    else:
        exp.run(data)

elif options.experiment == 1:
    print('Training model with 4 inputs and 4 outputs')
    data = Data(data_dir, dataset='BRATS', trim_and_downsample=False, modalities_to_load=['T1', 'T2', 'T2FLAIR'],
                normalize_volumes=True)
    data.load()

    input_modalities = ['T1', 'T2']
    output_weights = {'T2FLAIR': 1.0, 'concat': 1.0}
    exp = Experiment(input_modalities, output_weights, options.resultsdir, data, latent_dim=16, spatial_transformer=False)
    if options.checkpoint != None:
        exp.resume_from_checkpoint(data, options.checkpoint)
    else:
        exp.run(data)

elif options.experiment == 2:
    # TODO: Fix this
    print('Testing the model')
    data = Data(data_dir, dataset='BRATS', trim_and_downsample=False, modalities_to_load=['T1', 'T2', 'T1CE', 'T2FLAIR'], normalize_volumes=False)
    data.load()

    input_modalities = ['T1', 'T2', 'T1CE', 'T2FLAIR']
    output_weights = {'T1': 1.0, 'T2': 1.0, 'T1CE': 1.0, 'T2FLAIR': 1.0, 'concat': 1.0}
    exp = Experiment(input_modalities, output_weights, options.resultsdir, data, latent_dim=16, spatial_transformer=True)
    if options.checkpoint != None:
        exp.resume_from_checkpoint(data, options.checkpoint)
    else:
        exp.load_partial_model(folder=options.resultsdir + '/split0/', model_name='model', input_modalities=['T1', 'T2'], output_modality='T2FLAIR')
        exp.load_model(options.resultsdir + '/split0/', model_name='model')
        exp.run(data)

    exp.run(data)