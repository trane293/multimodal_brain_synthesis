from loader_multimodal import Data
from runner import Experiment
import optparse
import platform
import os
import itertools

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
                  default=0,
                  type='int',
                  help='Which experiment to perform'
                  )

parser.add_option('--n', '--exp-name',
                  dest="exp_name",
                  default='test',
                  type='str',
                  help='Name of experiment'
                  )

parser.add_option('--b', '--batch-size',
                  dest="batch_size",
                  default=5,
                  type='int',
                  help='Batch size to train with'
                  )

parser.add_option('--c', '--checkpoint',
                  dest="checkpoint",
                  default=None,
                  type='str',
                  help='Path to the latest checkpoint file to resume training from'
                  )


options, remainder = parser.parse_args()

if options.experiment == 0:
    print('Training model with 4 inputs and 4 outputs')
    all_modalities = ['T1', 'T2', 'T1CE', 'T2FLAIR']
    data = Data(data_dir, dataset='BRATS', trim_and_downsample=False, modalities_to_load=all_modalities, normalize_volumes=False)
    data.load()
    # T1W T2W T1C T2F
    #  0   1   1   1
    #  1   0   1   1
    #  1   1   0   1
    #  1   1   1   0
    for chosen in itertools.combinations(all_modalities, 3):
        print('Input Modality: {}'.format(chosen))
        for k in all_modalities:
            if k not in chosen:
                out_modality = k
        print('Output Modality: {}'.format(out_modality))

        model_prefix = '_'.join(chosen) + '-->' + out_modality
        input_modalities = list(chosen)

        output_weights = {out_modality: 1.0, 'concat': 1.0}
        exp = Experiment(input_modalities, output_weights, options.resultsdir, data, latent_dim=16, spatial_transformer=False)

        exp_name = options.exp_name + '_' + model_prefix

        if options.checkpoint != None:
            exp.resume_from_checkpoint(data, options.checkpoint)
        else:
            exp.run(data, exp_name=exp_name, batch_size=options.batch_size)

elif options.experiment == 1:
    print('Training model with 2 inputs and 1 outputs')
    data = Data(data_dir, dataset='BRATS', trim_and_downsample=False, modalities_to_load=['T1', 'T2', 'T2FLAIR'],
                normalize_volumes=False)
    data.load()

    input_modalities = ['T1', 'T2']
    output_weights = {'T2FLAIR': 1.0, 'concat': 1.0}
    exp = Experiment(input_modalities, output_weights, options.resultsdir, data, latent_dim=16, spatial_transformer=False)
    if options.checkpoint != None:
        exp.resume_from_checkpoint(data, options.checkpoint)
    else:
        exp.run(data, exp_name=options.exp_name, batch_size=options.batch_size)
