from dataload import DPRewriteDataset
from utils import decode_rewritten, EarlyStopping
from models.autoencoders.adept import ADePT, ADePTModelConfig
from models.autoencoders.custom import CustomModel_RNN, CustomModelConfig
import matplotlib.pyplot as plt
import time
import json
import copy
from tqdm import tqdm
from copy import deepcopy
import os
from abc import ABC, abstractmethod
from settings import Settings
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg as slin

os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'

import warnings
warnings.filterwarnings('ignore')

class Experiment(ABC):

    def __init__(self, ss:Settings):

        # General vars and directories
        self.seed = ss.args.seed
        self.mode = "pretrain"

        self.exp_output_dir = ss.exp_output_dir
        self.exp_dump_dir = ss.exp_dump_dir
        self.checkpoint_dir = ss.checkpoint_dir
        self.log_dir = os.path.join(self.exp_output_dir, ss.args.log_file_name)

        self.asset_dir = ss.args.asset_dir
        self.embed_dir_processed = ss.embed_dir_processed

        self.dataset_name = ss.args.dataset
        self.custom_train_path = ss.args.custom_train_path
        self.custom_valid_path = ss.args.custom_valid_path
        self.custom_test_path = ss.args.custom_test_path

        self.last_checkpoint_path = ss.args.last_checkpoint_path

        self.data_split_cutoff = ss.args.data_split_cutoff
        self.iteration_cutoff = ss.args.iteration_cutoff

        if ss.args.specified_device == None:
            self.device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(ss.args.specified_device)

        # Hyperparameters (general)
        self.model = ss.args.model.lower()
        self.model_type = ss.args.model_type
        self.max_seq_len = ss.args.max_seq_len
        self.optim_type = ss.args.optim_type
        self.epochs = ss.args.epochs
        self.batch_size = ss.args.batch_size
        self.learning_rate = ss.args.learning_rate
        self.weight_decay = ss.args.weight_decay
        self.early_stop = ss.args.early_stopping
        self.patience = ss.args.patience
        self.early_stopping = EarlyStopping(self.patience)

        self.optimizer = None
        self.enc_optimizer = None
        self.dec_optimizer = None
        self.mlp_optimizer = None
        self.loss = None

        # Hyperparameters (specific to models)
        self.train_teacher_forcing_ratio = ss.args.train_teacher_forcing_ratio
            # only for transformer-based models
        self.hidden_size = ss.args.hidden_size
        self.enc_out_size = ss.args.enc_out_size
            # general for experiments that add DP module after encoder outputs
        self.embed_type = ss.args.embed_type
        self.vocab_size = ss.args.vocab_size
            # for experiments with non-HF-based tokenizers
        self.embed_size = ss.args.embed_size

        self.custom_model_arguments = ss.args.custom_model_arguments

        # Private parameters (not all necessary, depending on DP module):
        self.private = ss.args.private
        self.epsilon = ss.args.epsilon
        self.delta = ss.args.delta
        self.clipping_constant = ss.args.clipping_constant
        self.norm_ord = ss.args.l_norm
        self.dp_module = ss.args.dp_module
        self.dp_mechanism = ss.args.dp_mechanism
        self.max_beam_size = ss.args.max_beam_size
        self.use_call_matrix = ss.args.use_call_matrix
        self.where_matrix = ss.args.where_matrix
        self.use_cfg = ss.args.use_cfg
        self.matrix_type = ss.args.matrix_type
        self.is_pruning = ss.args.is_pruning # whether to prune the call stack
        self.is_pruning_r2 = ss.args.is_pruning_r2 
        self.feature_matrix_prune_pos = ss.args.feature_matrix_prune_pos
        self.is_android = ss.args.is_android
        self.java_dataset_type = ss.args.java_dataset_type

        if self.dp_mechanism == 'gaussian' and self.norm_ord == 1:
            print(f"\n+++ WARNING: Using {self.dp_mechanism} noise with norm order {self.norm_ord}. +++\n")

        # Additional settings
        self.no_clipping = ss.args.no_clipping
        self.save_initial_model = ss.args.save_initial_model
        self.train_ratio = ss.args.train_ratio

        # General variables for experiments
        self.trainable_params = 0
        self.train_losses = []
        self.valid_losses = []
        self.best_score=dict()

        self.temp_train_file_original = None
        self.temp_train_file_preds = None
        self.temp_valid_file_original = None
        self.temp_valid_file_preds = None

        # Write configuration and various stats to json files for documentation
        self.stats = {}

        config = {key: value for key, value in ss.args.__dict__.items()
                  if not key.startswith('__') and not callable(key)}
        with open(os.path.join(self.exp_output_dir, 'config.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    @abstractmethod
    def _load_checkpoint(self):
        pass

    @abstractmethod
    def train_iteration(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def plot_learning_curve(self):
        pass

    @abstractmethod
    def run_experiment(self):
        pass


class PretrainExperiment(Experiment):
    def __init__(self, ss: Settings):
        super().__init__(ss)
        self.dataset = DPRewriteDataset(
                self.dataset_name, self.asset_dir, self.checkpoint_dir,
                self.max_seq_len, self.batch_size, mode=self.mode,
                embed_type=self.embed_type, train_ratio=self.train_ratio,
                embed_size=self.embed_size,
                embed_dir_processed=self.embed_dir_processed,
                vocab_size=self.vocab_size,
                model_type=self.model_type, private=self.private,
                data_split_cutoff=self.data_split_cutoff,
                custom_train_path=self.custom_train_path,
                custom_valid_path=self.custom_valid_path,
                custom_test_path=self.custom_test_path,
                is_android=self.is_android,java_dataset_type=self.java_dataset_type,
                asset_dir=self.asset_dir,is_pruning=self.is_pruning,is_pruning_r2=self.is_pruning_r2,
                )
        self.dataset.load_and_process()

        print('Initializing model...')
        self._init_model()

    def _init_model_config(self):
        # Setting the padding index
        if self.model_type == 'rnn':
            pad_idx = self.dataset.preprocessor.word2idx[
                self.dataset.preprocessor.PAD]

        self.pad_idx = pad_idx

        # Preparing the general model configuration
        general_config_dict = {
            'max_seq_len': self.max_seq_len, 'batch_size': self.batch_size,
            'mode': self.mode, 'device': self.device,
            'hidden_size': self.hidden_size,
            'enc_out_size': self.enc_out_size,
            'embed_size': self.embed_size, 'pad_idx': pad_idx,
            'private': self.private, 'epsilon': self.epsilon,
            'delta': self.delta, 'norm_ord': self.norm_ord,
            'clipping_constant': self.clipping_constant,
            'dp_mechanism': self.dp_mechanism,
            'experiment_output_dir': self.exp_output_dir,
            'use_call_matrix':self.use_call_matrix,'where_matrix':self.where_matrix,
            'use_cfg':self.use_cfg}

        # Preparing the specific model configuration class
        model_config = self._get_specific_model_config(general_config_dict)
        return model_config

    def _get_specific_model_config(self, general_config_dict):
        if self.model == 'adept':
            specific_config_dict = {
                'pretrained_embeddings': self.dataset.preprocessor.embeds,
                'vocab_size': self.dataset.preprocessor.vocab_size,
                'dp_module': self.dp_module,
                'no_clipping': self.no_clipping
                }
            specific_config = ADePTModelConfig(
                **general_config_dict, **specific_config_dict
                )
        elif self.model in ['custom_rnn', 'custom_transformer']:
            specific_config_dict = {
                'custom_config_list': self.custom_model_arguments}
            specific_config = CustomModelConfig(
                **general_config_dict, **specific_config_dict
                )
        else:
            raise NotImplementedError
        return specific_config

    def _get_model_type(self):
        if self.model == 'adept':
            model_type = ADePT
        elif self.model == 'custom_rnn':
            model_type = CustomModel_RNN
        else:
            raise NotImplementedError
        return model_type

    def create_feature_matrix(self):
        adj_matrix=copy.deepcopy(self.dataset.preprocessor.adj_matrix)

        # TODO:
        if self.is_pruning and self.feature_matrix_prune_pos==0:
            indices = torch.tensor(self.dataset.preprocessor.unique_paths_matrix, dtype=torch.long)==1
            adj_matrix[indices] = 1

        if self.matrix_type==0:
            feature_matrix = slin.expm(adj_matrix)
        else:
            feature_matrix = np.array(adj_matrix)
        
        if self.is_pruning and self.feature_matrix_prune_pos==1:
            indices = torch.tensor(self.dataset.preprocessor.unique_paths_matrix, dtype=torch.long)==1
            adj_matrix[indices] = 1
        return feature_matrix, np.array(adj_matrix)

    def _init_model(self):
        model_config = self._init_model_config()
        model_type = self._get_model_type()
        if self.dataset.preprocessor.adj_matrix is None:
            self.dataset.preprocessor._create_adj_matrix(self.dataset.train_data,self.dataset.valid_data)
        model = model_type(model_config, *self.create_feature_matrix())

        self.model = model.to(self.device)

        num_params = sum(p.numel()
                         for p in model.parameters() if p.requires_grad)
        print(f"Num parameters in model: {num_params,}")
        self.stats['num_params'] = num_params

        if self.optim_type == 'adam':
            optimizer = optim.Adam
        elif self.optim_type == 'sgd':
            optimizer = optim.SGD
        else:
            raise Exception('Incorrect optimizer type specified.')

        self.optimizer = optimizer(self.model.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=self.weight_decay)

        mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
        mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
        mem = mem_params + mem_bufs  # in bytes
        print("Estimated non-peak memory usage of model (MBs):", mem / 1000000)
        self.loss = nn.CrossEntropyLoss(ignore_index=self.pad_idx)

    def _load_checkpoint(self):
        '''
        Load existing checkpoint of a model and stats dict, if available.
        Stats dict only loaded if there is an existing checkpoint.
        '''
        try:
            if self.last_checkpoint_path is not None and self.last_checkpoint_path!='':
                mod_name = self.last_checkpoint_path
            else:
                mod_name = os.path.join(self.checkpoint_dir, 'checkpoint.pt')
            checkpoint = torch.load(mod_name, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loaded_epoch = checkpoint['checkpoint_epoch'] + 1
            # Restart training from the next epoch
            early_stopping_counter = checkpoint['checkpoint_early_stopping']
            print(f"Loaded model from epoch {loaded_epoch} with early stopping counter at {early_stopping_counter}.")

            try:
                stats_path = os.path.join(self.exp_output_dir, 'stats.json')
                with open(stats_path, 'r', encoding='utf-8') as f:
                    self.stats = json.load(f)
            except:
                print("Could not load existing stats dictionary.")

        except:
            print("Could not load checkpointed model, starting from scratch...")
            loaded_epoch = 0
            early_stopping_counter = 0

        return loaded_epoch, early_stopping_counter
        
    def train_iteration(self, epoch):
        epoch_loss = 0
        iter_size = len(self.dataset.train_iterator)

        self.model.train()
        for idx, batch in tqdm(enumerate(self.dataset.train_iterator)):
            if self.iteration_cutoff is not None and idx == self.iteration_cutoff:
                break

            if self.model_type == 'rnn':
                encoder_input_ids = batch[0]
                lengths = batch[1]
                encoder_input_ids = encoder_input_ids.to(self.device)
                inputs = {'input_ids': encoder_input_ids, 'lengths': lengths,
                          'teacher_forcing_ratio': self.train_teacher_forcing_ratio}
                tgt = deepcopy(encoder_input_ids)[:, 1:].reshape(-1)

            self.optimizer.zero_grad()
            loss = 0
            outputs = self.model(**inputs)
            loss = self.loss(outputs, tgt)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=1)

            self.optimizer.step()
            epoch_loss += loss.item()

            if idx == 0:
                preds = torch.max(outputs, dim=1).indices.view(
                        encoder_input_ids.shape[0],
                        encoder_input_ids.shape[1] - 1)

                decoded_text = decode_rewritten(
                        preds[0].unsqueeze(0),
                        self.dataset.preprocessor,
                        remove_special_tokens=False,
                        model_type=self.model_type)[0]
                original = decode_rewritten(
                        encoder_input_ids[0][1:].unsqueeze(0),
                        self.dataset.preprocessor,
                        remove_special_tokens=False,
                        model_type=self.model_type)[0]

                self.stats[f'sample_original_ep{epoch}_train'] = original
                self.stats[f'sample_pred_ep{epoch}_train'] = decoded_text

        return epoch_loss / iter_size

    def evaluate(self, epoch, final=False):
        epoch_loss = 0
        iter_size = len(self.dataset.valid_iterator)

        self.model.eval()
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(self.dataset.valid_iterator)):
                if self.iteration_cutoff is not None and idx == self.iteration_cutoff:
                    break

                if self.model_type == 'rnn':
                    encoder_input_ids = batch[0]
                    lengths = batch[1]
                    encoder_input_ids = encoder_input_ids.to(self.device)
                    inputs = {'input_ids': encoder_input_ids, 'lengths': lengths,
                              'teacher_forcing_ratio': 0.0}
                    tgt = deepcopy(encoder_input_ids)[:, 1:].reshape(-1)
                else:
                    encoder_input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    inputs = {'input_ids': encoder_input_ids,
                              'attention_mask': attention_mask}
                    tgt = deepcopy(encoder_input_ids)[:, 1:].reshape(-1)

                if self.two_optimizers:
                    self.enc_optimizer.zero_grad()
                    self.dec_optimizer.zero_grad()
                    self.mlp_optimizer.zero_grad()
                else:
                    self.optimizer.zero_grad()

                loss = 0

                outputs = self.model(**inputs)

                loss = self.loss(outputs, tgt)
                loss = loss.item()

                epoch_loss += loss

                if idx == 0 and not final:
                    preds = torch.max(outputs, dim=1).indices.view(
                            encoder_input_ids.shape[0],
                            encoder_input_ids.shape[1] - 1)

                    decoded_text = decode_rewritten(
                            preds[0].unsqueeze(0),
                            self.dataset.preprocessor,
                            remove_special_tokens=False,
                            model_type=self.model_type)[0]
                    original = decode_rewritten(
                            encoder_input_ids[0][1:].unsqueeze(0),
                            self.dataset.preprocessor,
                            remove_special_tokens=False,
                            model_type=self.model_type)[0]

                    '''print("VALID ORIGINAL: ", original)
                    print("VALID PRED: ", decoded_text)'''
                    self.stats[f'sample_original_ep{epoch}_valid'] = original
                    self.stats[f'sample_pred_ep{epoch}_valid'] = decoded_text

                if final:
                    preds = torch.max(outputs, dim=1).indices.view(
                            encoder_input_ids.shape[0],
                            encoder_input_ids.shape[1] - 1)

                    decoded_text = decode_rewritten(
                            preds, self.dataset.preprocessor,
                            remove_special_tokens=True,
                            model_type=self.model_type)
                    original = decode_rewritten(
                            encoder_input_ids, self.dataset.preprocessor,
                            remove_special_tokens=True,
                            model_type=self.model_type)

                    for batch_idx in range(len(decoded_text)):
                        with open(self.temp_valid_file_preds, 'a', encoding='utf-8') as f:
                            f.write(decoded_text[batch_idx])
                            f.write('\n')
                        with open(self.temp_valid_file_original, 'a', encoding='utf-8') as f:
                            f.write(original[batch_idx])
                            f.write('\n')

        return epoch_loss / iter_size

    def train(self, loaded_epoch=0, early_stopping_counter=0):

        self.early_stopping.counter = early_stopping_counter


        for epoch in range(loaded_epoch, self.epochs):

            start_time = time.time()
            train_loss = self.train_iteration(epoch)
            valid_loss = 0#self.evaluate(epoch, final=False)
            end_time = time.time()
            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)

            self.plot_learning_curve()

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f}')
            
            with open(self.log_dir, 'a', encoding='utf-8') as f:
                f.write(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n')
                f.write(f'\tTrain Loss: {train_loss:.3f}\n')

            # Saving checkpoint
            early_stop = self._save_checkpoint_and_early_stopping(
                epoch, valid_loss, early_stop=self.early_stop)
            if early_stop:
                break

            # Updating stats dictionary
            self.stats[f'pretrain_epoch_mins_{epoch}'] = epoch_mins
            self.stats[f'pretrain_epoch_secs_{epoch}'] = epoch_secs
            self.stats[f'pretrain_train_loss_{epoch}'] = train_loss
            self.stats[f'pretrain_valid_loss_{epoch}'] = valid_loss

            # Saving stats dictionary
            self._save_stats_dict()
            if self.is_android:
                if (epoch+1)%5==0 or (epoch+1)==self.epochs:
                    '''self._save_checkpoint_and_early_stopping(
                        epoch, valid_loss, early_stop=self.early_stop,save_per_epoch=True)'''
                    for beam_size in range(1,self.max_beam_size+1):
                        self.evaluate(beam_size, cur_epoch=epoch+1, final=False)
                    score_keys = [0.001,0.005,0.01,0.05]
                    score_str="Current best f1 score: \n"
                    score_str+=f"Epoch: {self.best_score['epoch']}, Beam size: {self.best_score['beam_size']}\n"
                    for i,eval_score in enumerate(self.best_score['scores']):
                        score_str+=f"Threshold: {score_keys[i]}, F1 score: {eval_score:.3f}; "
                    with open(self.log_dir, "a") as f:
                        f.write(score_str+'\n')
            else:
                if (epoch+1)%5==0 or (epoch+1)==self.epochs:
                    self._save_checkpoint_and_early_stopping(
                    epoch, valid_loss, early_stop=self.early_stop,save_per_epoch=True)
                    for beam_size in range(1,self.max_beam_size+1):
                        self.evaluate(beam_size, cur_epoch=epoch+1, final=False)
                    score_keys = [0.001,0.005,0.01,0.05]
                    score_str="Current best f1 score: \n"
                    score_str+=f"Epoch: {self.best_score['epoch']}, Beam size: {self.best_score['beam_size']}\n"
                    for i,eval_score in enumerate(self.best_score['scores']):
                        score_str+=f"Threshold: {score_keys[i]}, F1 score: {eval_score:.3f}; "
                    with open(self.log_dir, "a") as f:
                        f.write(score_str+'\n')
        # Saving stats dictionary one last time
        self._save_stats_dict()

    def _save_stats_dict(self):
        with open(os.path.join(self.exp_output_dir, 'stats.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=4)

    def _save_checkpoint_and_early_stopping(self, epoch, valid_loss, early_stop=True,save_per_epoch=False):
        if save_per_epoch:
            checkpoint_name = os.path.join(self.checkpoint_dir, f'checkpoint_{epoch}.pt')
        else:
            checkpoint_name = os.path.join(self.checkpoint_dir, f'checkpoint.pt')

        checkpoint_dict = {
            'checkpoint_epoch': epoch,
            'checkpoint_early_stopping': self.early_stopping.counter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }

        # Early stopping
        if early_stop:
            self.early_stopping(valid_loss, checkpoint_dict, checkpoint_name)
            return self.early_stopping.early_stop
        else:
            torch.save(checkpoint_dict, checkpoint_name)
            return False

    def _set_up_evaluation_files(self):
        self.temp_valid_file_original = os.path.join(
                self.exp_output_dir, 'temp_valid_original.txt')
        self.temp_valid_file_preds = os.path.join(
                self.exp_output_dir, 'temp_valid_preds.txt')
        with open(self.temp_valid_file_original, 'w', encoding='utf-8') as f:
            f.write('valid original\n')
        with open(self.temp_valid_file_preds, 'w', encoding='utf-8') as f:
            f.write('valid preds\n')

    def _get_refs_and_hyps(self, preds_file, original_file):
        with open(preds_file, 'r', encoding='utf-8') as f:
            hyps = [x.strip() for x in f]
            hyps = hyps[1:]
        with open(original_file, 'r', encoding='utf-8') as f:
            refs = [x.strip() for x in f]
            refs = [refs[1:]]
        return hyps, refs

    def plot_learning_curve(self):
        '''
        Result png figures are saved in the log directory.
        '''
        fig, ax = plt.subplots(num=1, clear=True)
        fig.suptitle('Model Learning Curve')

        epochs = list(range(len(self.train_losses)))
        ax.plot(epochs, self.train_losses, 'o-', markersize=2, color='b',
                label='Train')
        ax.plot(epochs, self.valid_losses, 'o-', markersize=2, color='c',
                label='Validation')
        ax.set(xlabel='Epoch', ylabel='Pretrain Loss')
        ax.legend()

        plt.savefig(os.path.join(self.exp_output_dir, 'learning_curve.png'))

    def run_experiment(self):
        # Load an existing model checkpoint, if available
        loaded_epoch, early_stopping_counter = self._load_checkpoint()

        if self.save_initial_model and loaded_epoch == 0:
            # For convenient comparison of non-pretrained models
            print("Saving initial checkpoint of model...")
            self._save_checkpoint_and_early_stopping(-1, np.inf,
                                                     early_stop=False)

        # Setting up files for later evaluation of outputs
        self._set_up_evaluation_files()

        self.train(loaded_epoch=loaded_epoch,
                   early_stopping_counter=early_stopping_counter)
        
    def initialize_model(self):
        # Load an existing model checkpoint, if available
        loaded_epoch, early_stopping_counter = self._load_checkpoint()

        if self.save_initial_model and loaded_epoch == 0:
            # For convenient comparison of non-pretrained models
            print("Saving initial checkpoint of model...")
            self._save_checkpoint_and_early_stopping(-1, np.inf,
                                                     early_stop=False)

        # Setting up files for later evaluation of outputs
        self._set_up_evaluation_files()

    def evaluate(self, cur_beam_size, cur_epoch=0, final=False):
        def _computeRE(union_ground_truth,union_ground_pred):
                # compute accuracy for nodes that appear in at least one
                # tree; this is not a great metric, as it does not consider
                # nodes that do *not* appear in any trees.
                l1 = 0
                l1_ground = 0
                for long_id in union_ground_truth:
                    ground_freq = union_ground_truth[long_id]
                    l1_ground += ground_freq
                for long_id in union_ground_pred:
                    ground_freq = union_ground_pred[long_id]
                    l1 += ground_freq
                if l1_ground == 0:
                    return 0
                else:
                    return(abs(1-l1/l1_ground))
            
        def _bms_metric_test(union_ground_truth,union_ground_pred,threshold=0.5,beam_size=7,limit=None):
            limit = self.data_split_cutoff*threshold if limit is None else limit
            hot_truth_keys = {k:v for k, v in union_ground_truth.items() if v > limit}
            hot_pred_keys = {k:v for k, v in union_ground_pred.items() if v > limit}
            
            Re = _computeRE(union_ground_truth,union_ground_pred)
            #REforEHH = _computeREforEHH(hot_truth_keys,hot_pred_keys)
            REforEHH = _computeRE(hot_truth_keys,hot_pred_keys)
            
            result_str = f"Beam Size: {beam_size}, Threshold: {threshold}, limit:{limit}, Error Hot: {REforEHH:.3f}\n"
            inter = set(hot_truth_keys.keys()).intersection(set(hot_pred_keys.keys()))
            if len(set(hot_truth_keys.keys())) == 0:
                recall = 0
            else:
                recall = len(inter)/len(set(hot_truth_keys.keys()))
            if len(set(hot_pred_keys.keys())) == 0:
                precision = 0
            else:
                precision = len(inter)/len(set(hot_pred_keys.keys()))
            if precision + recall == 0:
                f1_score = 0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)
            result_str += f"Precision: {precision:.3f}, "
            result_str += f"Recall: {recall:.3f}, "
            result_str += f"F1 score: {f1_score:.3f}\n"
            
            print(result_str)
            with open(self.log_dir, "a") as f:
                f.write(result_str)

            return f1_score


        # initialize the ground truth and prediction dictionaries
        union_ground_truth=dict()
        union_ground_pred=dict()

        start_time = time.time()
        for batch_idx in range(len(self.dataset.original_test_data)):
            if self.dataset.original_test_data[batch_idx] not in union_ground_truth.keys():
                union_ground_truth[self.dataset.original_test_data[batch_idx]] = 1
            else:
                union_ground_truth[self.dataset.original_test_data[batch_idx]] += 1
        
        self.model.eval()

        with torch.no_grad():
            for data_iter in [self.dataset.test_iterator]:
                print("------------------")
                for idx, batch in tqdm(enumerate(data_iter)):
                    if self.model_type == 'rnn':
                        encoder_input_ids = batch[0]
                        lengths = batch[1]
                        encoder_input_ids = encoder_input_ids.to(self.device)
                        inputs = {'input_ids': encoder_input_ids, 'lengths': lengths,
                                'teacher_forcing_ratio': self.train_teacher_forcing_ratio}
                        tgt = deepcopy(encoder_input_ids)[:, 1:].reshape(-1)
                    else:
                        encoder_input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        inputs = {'input_ids': encoder_input_ids,
                                    'attention_mask': attention_mask}
                        tgt = deepcopy(encoder_input_ids)[:, 1:].reshape(-1)

                    self.optimizer.zero_grad()

                    if True:                        
                        preds,scores = self.model.decode(beam_size=cur_beam_size,tokenizer=self.dataset.preprocessor.tokenizer,**inputs)
                        for batch_idx in range(0,len(encoder_input_ids)):
                            softmax_scores = F.softmax(scores[batch_idx]) if cur_beam_size!=1 else [1]
                            decoded_text = decode_rewritten(
                                preds[batch_idx], self.dataset.preprocessor,
                                remove_special_tokens=True,
                                model_type=self.model_type,is_pruning=self.is_pruning)
                            for beam_seq_idx in range(len(preds[batch_idx])):
                                if decoded_text[beam_seq_idx] not in union_ground_pred.keys():
                                    union_ground_pred[decoded_text[beam_seq_idx]] = softmax_scores[beam_seq_idx]
                                else:
                                    union_ground_pred[decoded_text[beam_seq_idx]] += softmax_scores[beam_seq_idx]
            end_time = time.time()
            used_time = end_time - start_time       
            print(f"*** Call-chain Analysis Result for batik ***")
            with open(self.log_dir, "a") as f:
                f.write(f"*** Call-chain Analysis Result for batik, used time:{used_time} ***\n")
            scores=[]
            for threshold in [0.001,0.005,0.01,0.05]:
                limit = len(self.dataset.original_test_data)*threshold
                scores.append(_bms_metric_test(union_ground_truth,union_ground_pred,threshold=threshold,beam_size=cur_beam_size,limit=limit))
            total_f1_score = sum(scores)/len(scores)
            
            if not final:
                if len(self.best_score)==0:
                    self.best_score['scores'] = scores
                    self.best_score["epoch"]=cur_epoch
                    self.best_score["beam_size"]=cur_beam_size
                else:
                    if total_f1_score > sum(self.best_score['scores'])/len(self.best_score['scores']):
                        self.best_score['scores'] = scores
                        self.best_score["epoch"]=cur_epoch
                        self.best_score["beam_size"]=cur_beam_size
        return 
    