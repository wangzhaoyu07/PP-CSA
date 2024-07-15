import os
import pathlib
import time
import argparse
import re
from utils import get_model_type
import pdb

project_root = os.path.join(pathlib.Path(__file__).parent.resolve())


def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Expected boolean value.')


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)


def parse_arguments():
    argparser = argparse.ArgumentParser()

    # Main parameters
    argparser.add_argument("--dataset", type=str, required=True,
                           help='Which dataset will be used for training the autoencoder. ')
    argparser.add_argument("--model", type=str, default='adept',
                           help="Model to run")

    # optional, general params: meta params for running experiments
    argparser.add_argument("--name", type=str, default=None,
                           help='The experiment name. Defaults to the current timestamp.')
    argparser.add_argument("--seed", type=int, default=12345)

    # optional, general params: directories
    argparser.add_argument("--output_dir", type=str, default=os.path.join(project_root, 'results'),
                           help='Where stats & logs will be saved (a subfolder will be created for '
                                'the experiment). Defaults to <project_root>/results')
    argparser.add_argument("--dump_dir", type=str, default=None,
                           help='Where stuff that might need much storage (e.g., model checkpoints) '
                                'will be saved. Defaults to output_dir.')

    argparser.add_argument("--asset_dir", type=str, default=os.path.join(project_root, 'assets'),
                           help='Where to look for assets like data sets & embeddings. '
                                'Defaults to <project_root>/assets. '
                                'For data sets, this can be overwritten with the data_dir argument.')
    argparser.add_argument("--data_dir", type=str, default=None,
                           help='Where to look for data sets. Defaults to asset_dir')

    argparser.add_argument("--custom_train_path", type=str, default=None, help='Where to look for a custom datasets (train partition).')
    argparser.add_argument("--custom_valid_path", type=str, default=None, help='Where to look for a custom datasets (optional validation partition).')
    argparser.add_argument("--custom_test_path", type=str, default=None, help='Where to look for a custom datasets (optional test partition).')

    # optional, general params: datasets
    argparser.add_argument("--train_ratio", type=float, default=0.8,
                           help='Training dataset size ratio for train/validation split. Not used if custom dataset specified with a path for the validation set.')
    argparser.add_argument("--last_checkpoint_path", type=str, default='',
                           help='Global path of the checkpoint that should be used, which will be used to resume training. ')
    argparser.add_argument("--data_split_cutoff", type=int, default=None, help='If not "None", will use a subset of the "train" data split based on the specified int value (e.g. 10000 to only use the first 10000 data points). Note: This is applied before any train-validation splits, in case validation data is not present.')
    argparser.add_argument("--iteration_cutoff", type=int, default=None, help='If not "None", will only run training and validation for the specified amount of iterations every epoch. ')

    # optional, general params: hyperparams
    argparser.add_argument("--epochs", type=int, default=5)
    argparser.add_argument("--batch_size", type=int, default=1)
    argparser.add_argument("--learning_rate", type=float, default=0.001)
    argparser.add_argument("--weight_decay", type=float, default=0.00)
    argparser.add_argument("--early_stopping", type=str2bool, nargs='?',
                           const=True, default=False)
    argparser.add_argument("--patience", type=int, default=20)
    argparser.add_argument("--optim_type", type=str, default='adam',
                           help='sgd or adam')
    argparser.add_argument("--hidden_size", type=int, default=768)
    argparser.add_argument("--enc_out_size", type=int, default=128,
                           help='Specific to RNN models.')
    argparser.add_argument("--vocab_size", type=int, default=99)
    argparser.add_argument("--max_seq_len", type=int, default=512)
    argparser.add_argument("--embed_size", type=int, default=300)
    argparser.add_argument("--embed_type", type=str, default='none',
                           help='"glove" or "word2vec"')
    argparser.add_argument("--train_teacher_forcing_ratio", type=float,
                           default=0.0, help='For RNN-based models.')
    argparser.add_argument("--private", type=str2bool, nargs='?',
                           const=True, default=False,
                           help='If privatization should be applied during pre-training')
    argparser.add_argument("--epsilon", type=float, default=1)
    argparser.add_argument("--delta", type=float, default=1e-5)
    argparser.add_argument("--clipping_constant", type=float, default=1.)
    argparser.add_argument("--save_initial_model", type=str2bool, nargs='?', const=True, default=False, help='Whether to save a checkpoint of the model before starting the training procedure.')
    argparser.add_argument("--dp_mechanism", type=str, default='laplace', help='laplace or gaussian.'
                            'Has no effect as of now.')
    argparser.add_argument("--dp_module", type=str, default='clip_norm', help='The type of DP module to be applied, specific to certain autoencoder models. Relevant arguments to be specified for each specific module.')
    argparser.add_argument("--l_norm", type=int, default=2, help='Pass 2 for L2 norm, 1 for L1 norm.')
    argparser.add_argument("--no_clipping", type=str2bool, nargs='?', const=True, default=False, help='Whether or not to clip encoder hidden states in the non-private setting.')
    argparser.add_argument("--custom_model_arguments", nargs='*', help='Additional optional arguments for a custom model, no upper limit on the number.')
    argparser.add_argument("--specified_device", type=str, default=None,
                           help='Specify a device to use, e.g. "cuda" or "cpu".')
    argparser.add_argument("--max_beam_size", type=int, default=3, help='the max beam size')
    argparser.add_argument("--log_file_name", type=str, default='log.txt', help='the log file name')
    argparser.add_argument("--use_call_matrix", type=str2bool, default=False, help='whether to add additional feature')
    argparser.add_argument("--where_matrix", type=int, default=1, help='where to use matrix')
    argparser.add_argument("--use_cfg", type=str2bool, default=True, help='whether to use cfg to guide beam search')
    argparser.add_argument("--matrix_type", type=int, default=0, help='which matrix to use')
    argparser.add_argument("--is_pruning", type=str2bool, default=False, help='whether to prune call stacks')
    argparser.add_argument("--feature_matrix_prune_pos", type=int, default=1, help='where to prune call stacks')
    argparser.add_argument("--is_android", type=str2bool, default=False, help='whether the call stack dataset is android')
    argparser.add_argument("--java_dataset_type", type=int, default=0, help='the type of java dataset')
    argparser.add_argument("--is_pruning_r2", type=str2bool, default=False, help='Whether to perform lengthy stack cutoff')
    args = argparser.parse_args()

    return args


class Settings(object):
    '''
    Configuration for the project.
    '''
    def __init__(self, args):
        # args, e.g. the output of settings.parse_arguments()
        self.args = args

        now = time.localtime()
        self.args.formatted_now = f'{now[0]}-{now[1]}-{now[2]}--{now[3]:02d}-{now[4]:02d}-{now[5]:02d}'

        ## Determining model type
        self.args.model_type = get_model_type(self.args.model)

        if self.args.name is None or self.args.name == '':
            self.args.name = self.args.formatted_now
        if self.args.dump_dir is None:
            self.args.dump_dir = self.args.output_dir
        if self.args.data_dir is None:
            self.args.data_dir = self.args.asset_dir

        self.exp_output_dir = os.path.join(self.args.output_dir, self.args.name)
        self.exp_dump_dir = os.path.join(self.args.dump_dir, self.args.name)
        self.checkpoint_dir = os.path.join(self.exp_dump_dir, 'checkpoints')
        self.embed_dir_processed = None

    def make_dirs(self):
        for d in [
            self.args.output_dir,
            self.args.dump_dir,
            self.args.asset_dir,
            self.args.data_dir,
            self.exp_output_dir,
            self.exp_dump_dir,
            self.checkpoint_dir
            ]:
            if not os.path.exists(d):
                os.makedirs(d)

        if self.args.model_type == 'rnn':
            self.embed_dir_processed = os.path.join(self.args.asset_dir,
                                                    'embeds')
            if not os.path.exists(self.embed_dir_processed):
                os.makedirs(self.embed_dir_processed)
