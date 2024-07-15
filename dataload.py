import os
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from datasets import DatasetDict
from preprocessing import Preprocessor_for_RNN
import random
import copy

class DPRewriteDataset(object):
    def __init__(self, dataset_name, data_dir, checkpoint_dir, max_seq_len,
                 batch_size, mode='pretrain', train_ratio=0.9,
                 embed_type='glove', embed_size=300, embed_dir_processed=None,
                 vocab_size=None, model_type='rnn', private=False,
                 length_threshold=None,
                 data_split_cutoff=None,java_dataset_type=0,
                 last_checkpoint_path=False,
                 custom_train_path=None, custom_valid_path=None,
                 custom_test_path=None,
                 is_stack=True, asset_dir=None, is_pruning=False,is_pruning_r2=False,is_android=False
                ):
        self.dataset_name = dataset_name
        self.is_android = is_android
        self.java_dataset_type = java_dataset_type


        # main directory where data is stored (all modes; e.g. imdb, yelp)
        self.data_dir = data_dir

        self.checkpoint_dir = checkpoint_dir
        self.last_checkpoint_path = last_checkpoint_path

        # vocabulary and embeddings directory (renamed from 'in_dir')
        # used after processing vectors from below 'vec_model_dir'
        self.embed_dir_processed = embed_dir_processed

        self.mode = mode
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.model_type = model_type
        self.embed_type = embed_type
        self.length_threshold = length_threshold
        self.train_ratio = train_ratio
        self.data_split_cutoff = data_split_cutoff
        self.is_stack = is_stack

        self.train_data = None
        self.valid_data = None
        self.test_data = None

        self.sample_size = None
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None

        self.custom_train_path = custom_train_path
        self.custom_valid_path = custom_valid_path
        self.custom_test_path = custom_test_path

        self.is_pruning_r2 = is_pruning_r2
        self.top_num1 = None
        if self.is_pruning_r2:
            self.top_num1 = self._load_threshold()

        if model_type == 'rnn':
            if embed_dir_processed is None:
                raise Exception("Please specify 'embed_dir_processed' for RNN-based models.")
            self.preprocessor = Preprocessor_for_RNN(
                    embed_dir_processed, vocab_size=vocab_size, embed_type=embed_type,
                    embed_size=embed_size, checkpoint_dir=checkpoint_dir,
                    max_seq_len=max_seq_len, batch_size=batch_size,java_dataset_type=java_dataset_type,
                    mode=mode, asset_dir=asset_dir,
                    is_pruning=is_pruning,dataset_name=dataset_name,is_android=is_android)

        self.private = private
        self.shuffle = True

    def load_and_process(self):
        self.load()
        self.process()
        self.prepare_dataloader()

    def load(self, subset=None):
        '''
        Description
        -----------
        Prepares a 'Dataset' object, with features consisting of 'text' and
        'label'.

        Parameters
        ----------
        subset : ``int``, Don't load the full dataset, only up to a certain
                 index.
        '''
        print("Preparing custom dataset...")
        self._load_custom()

    def process(self):
        '''
        Description
        -----------
        Applies `_process_split()` method to each data split.
        '''
        self.train_data = self._process_split(self.train_data,
                                              train_split=True)
        if self.valid_data is not None:
            self.valid_data = self._process_split(self.valid_data)
        if self.test_data is not None:
            self.test_data = self._process_split(self.test_data)

    def prepare_dataloader(self):
        self.train_iterator = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=self.shuffle)
        self.sample_size = len(self.train_data)
        print('Num training:', self.sample_size)

        if self.valid_data is not None:
            self.valid_iterator = DataLoader(
                self.valid_data, batch_size=self.batch_size,
                shuffle=self.shuffle)
            print('Num validation:', len(self.valid_data))

        if self.test_data is not None:
            self.test_iterator = DataLoader(
                self.test_data, batch_size=self.batch_size,
                shuffle=self.shuffle)
            print('Num test:', len(self.test_data))

    def _load_custom(self):
        if self.custom_train_path is not None:
            self.train_data = self._load_custom_split(self.custom_train_path)
            #tmp_train_data = copy.deepcopy(self.train_data)
        else:
            raise Exception(
                f"{self.dataset_name} not in currently prepared datasets, "
                f"but 'custom_train_path' is None. Please either specify a "
                f"dataset name among existing datasets, or add a custom "
                f"dataset path.")

        if self.custom_valid_path is not None and \
        self.custom_valid_path.lower() != 'none':
            self.valid_data = self._load_custom_split(self.custom_valid_path)
        else:
            # If no validation path specified, make a split from the
            # training set
            if self.java_dataset_type==1:
                random.seed(0)
                total_samples = len(self.train_data)
                random_indices = random.choices(range(total_samples), k=self.data_split_cutoff)
                self.valid_data = self.train_data[random_indices]
            else:
                data_split = self.train_data.train_test_split(
                    test_size=(1-self.train_ratio),seed=0)
                self.train_data = data_split['train']
                self.valid_data = data_split['test']

        if self.custom_test_path is not None and \
        self.custom_test_path.lower() != 'none':
            self.test_data = self._load_custom_split(self.custom_test_path)
        else:
            """if self.is_android: # TODO:
                '''random.seed(0)
                self.test_data = self._load_custom_test_data(self.custom_train_path)'''
                total_samples = len(self.train_data)
                sample_num_per_client = 10
                random_indices = [random.randint(i  * sample_num_per_client, (i+1) * sample_num_per_client - 1) for i in range(0, total_samples//sample_num_per_client)]#list(range(0,total_samples,10))#[random.randint((i - 1) * 10, i * 10 - 1) for i in range(1, 1001)]random.choices(range(total_samples), k=self.data_split_cutoff)
                self.test_data = self.train_data.select(random_indices)
                self.original_test_data = list(self.test_data['text'])
            else:"""
            '''self.test_data = self._load_custom_test_data(self.custom_train_path)
            self.original_test_data = list(self.test_data['text'])'''
            random.seed(0)
            self.test_data = self._load_custom_test_data(self.custom_train_path)
            '''total_samples = len(self.train_data)
            random_indices = random.sample(range(total_samples), k=self.data_split_cutoff)
            self.test_data = self.train_data.select(random_indices)'''
            self.original_test_data = list(self.test_data['text'])


    def _load_threshold(self):
        f_dir = os.path.join(self.data_dir, self.dataset_name)
        with open(os.path.join(f_dir,'threshold'), 'r') as f:
            threshold = int(f.readline().strip())
        return threshold
    
    def _load_custom_split(self, path):
        ''''''
        random.seed(0)
        if self.is_android:
            with open(path, 'r') as f:
                f_paths = f.readlines()
            traces = []
            f_dir = os.path.join(self.data_dir, self.dataset_name)
            f_count=0
            total_file_num = len(f_paths)
            while f_count < self.data_split_cutoff:
                with open(os.path.join(f_dir,f_paths[f_count%total_file_num].strip()), 'r') as f:
                    data = f.readlines()
                    traces+=self._restore_sequences_for_android(data)
                f_count+=1
            df_exploded = pd.DataFrame({'sub_trace': traces})
        else:
            df = pd.read_csv(path, header=None, names=['trace'])
            if self.data_split_cutoff is not None:
                df = df.iloc[:self.data_split_cutoff]
            if self.java_dataset_type==0:
                df['sub_trace'] = df['trace'].apply(self._restore_sequences)
                df_sequences = df.explode('sub_trace')
                df_exploded = df_sequences.reset_index(drop=True).drop('trace', axis=1)
            else:
                df_exploded = df

        df_exploded.columns=["label"]
        df_exploded['text'] = np.array([None] * len(df_exploded))
        df_dict = {'train': df_exploded}
        data = DatasetDict({key: Dataset.from_pandas(df) for key, df in df_dict.items()})
        
        data = data['train']
        if np.all(np.array(data['text']) == None):
            # If there is only one column in the CSV file, then the
            # second column in the dataset will only have None, hence
            # need to remove it and rename the first column
            data = data.remove_columns("text")
            data = data.rename_column("label", "text")
            data = data.add_column("label", np.zeros(len(data)))
        return data

    def _load_custom_test_data(self, path):
        random.seed(0)
        if self.is_android:
            with open(path, 'r') as f:
                f_paths = f.readlines()
            traces = []
            f_dir = os.path.join(self.data_dir, self.dataset_name)
            f_count=0
            total_file_num = len(f_paths)
            while len(traces) < self.data_split_cutoff:
                with open(os.path.join(f_dir,f_paths[f_count%total_file_num].strip()), 'r') as f:
                    data = f.readlines()
                    trace = self._restore_sequences_for_android_for_test(data)
                    if len(trace)>0:
                        traces.append(trace)
                f_count+=1
            df_exploded = pd.DataFrame({'sub_trace': traces})
        else:
            df = pd.read_csv(path, header=None, names=['trace'])
            if self.data_split_cutoff is not None:
                df = df.iloc[:self.data_split_cutoff]
            if self.is_stack:
                df['sub_trace'] = df['trace'].apply(self._restore_sequences_for_test)
            else:
                df['sub_trace'] = df['trace'].apply(self._restore_sequences)
            df_sequences = df.explode('sub_trace').dropna()
            df_exploded = df_sequences.reset_index(drop=True).drop('trace', axis=1)
        df_exploded.columns=["label"]
        df_exploded['text'] = np.array([None] * len(df_exploded))
        df_dict = {'train': df_exploded}
        data = DatasetDict({key: Dataset.from_pandas(df) for key, df in df_dict.items()})
        ''''''
        
        data = data['train']
        if np.all(np.array(data['text']) == None):
            # If there is only one column in the CSV file, then the
            # second column in the dataset will only have None, hence
            # need to remove it and rename the first column
            data = data.remove_columns("text")
            data = data.rename_column("label", "text")
            data = data.add_column("label", np.zeros(len(data)))
        return data

    def _process_split(self, data, train_split=False):
        '''
        Description
        -----------
        Carries out preprocessing on the loaded dataset (additional sharding
        process for larger datasets).
        Resulting preprocessed dataset:
            len(data): length of dataset split
            data[i][0]: torch tensor of max_seq_len
            data[i][1]: length of tensor
            data[i][2]: label string

        Parameters
        ----------
        data : ``Dataset``, Loaded dataset object.
        '''
        # Optionally removing parts of the dataset, where the token count is
        # lower than a given threshold (based on whitespace split)
        if self.length_threshold is not None and \
                str(self.length_threshold).lower() != 'none':
            data = data.filter(
                lambda example: len(
                    example['text'].split()) <= self.length_threshold if example['text'] is not None else False)

        threshold = 2000000
        if len(data) > threshold:
            # Preprocessing for large datasets
            num_shards = 4
            new_shards = []
            print(f"Dataset very large, splitting preprocessing into "
                  f"{num_shards} shards.")
            for idx in range(num_shards):
                if idx > 0:
                    first_shard = False
                else:
                    first_shard = True
                new_shard = self.preprocessor.process_data(
                    data.shard(num_shards=num_shards, index=idx),
                    first_shard=first_shard, train_split=train_split)
                new_shards += new_shard
            data = new_shards
        else:
            data = self.preprocessor.process_data(
                data, train_split=train_split, first_shard=True)
        return data

    
    def _restore_sequences(self,s):
        stack = []
        sequences = set()
        actions = s.split(";")
        for action in actions:
            r2_threshold = self.top_num1 if self.is_pruning_r2 else 50
            if action.startswith("[Enter]"):
                item = action[7:]
                stack.append(item)
                if len(stack)>r2_threshold:
                    continue
                if stack:
                    if self.model_type == 'rnn':
                        sequences.add(';'.join(stack))
            elif action.startswith("[Exit]"):
                item = action[6:]
                if stack and stack[-1] == item:
                    stack.pop()
                else:
                    return []
            else:
                return []
        return sorted(list(sequences))
    
    def _restore_sequences_for_android(self,s):
        stack = []
        sequences = set()
        actions = s#.split(";")
        for action in actions:
            action = action.strip()
            r2_threshold = self.top_num1 if self.is_pruning_r2 else 50
            if len(sequences)>=50:
                break
            if action.startswith("E-"):
                item = action[2:]
                stack.append(item)
                if len(stack)>r2_threshold:
                    continue
                if stack:
                    if self.model_type == 'rnn':
                        sequences.add(';'.join(stack))
                    else:
                        sequences.add(''.join(stack))
            elif action.startswith("X-"):
                item = action[2:]
                if stack and stack[-1] == item:
                    stack.pop()
                else:
                    return []
            else:
                return []
        if self.is_android:
            if len(list(sequences))<10:
                return sorted(list(sequences))
            else:
                return sorted(random.sample(sorted(list(sequences)),10))
    
    def _restore_sequences_for_test(self,s):
        stack = []
        sequences = set()
        actions = s.split(";")
        r2_threshold = self.top_num1 if self.is_pruning_r2 else None
        for action in actions:
            if len(sequences)>=50:
                break
            if action.startswith("[Enter]"):
                item = action[7:]
                stack.append(item)
                if stack:
                    if self.model_type == 'rnn':
                        sequences.add(';'.join(stack))
                    else:
                        sequences.add(''.join(stack))
            elif action.startswith("[Exit]"):
                item = action[6:]
                if stack and stack[-1] == item:
                    stack.pop()
                else:
                    return []
            else:
                return []
        tmp_ls= random.choice(sorted(list(sequences)))
        if self.is_pruning_r2:
            if len(tmp_ls.split(";"))<=r2_threshold:
                return tmp_ls
            else:
                return []
        return  tmp_ls
    
    def _restore_sequences_for_android_for_test(self,s):
        stack = []
        sequences = set()
        actions = s
        r2_threshold = self.top_num1 if self.is_pruning_r2 else None
        for action in actions:
            if len(sequences)>=50:
                break
            action = action.strip()
            if action.startswith("E-"):
                item = action[2:]
                stack.append(item)
                if stack:
                    if self.model_type == 'rnn':
                        sequences.add(';'.join(stack))
                    else:
                        sequences.add(''.join(stack))
            elif action.startswith("X-"):
                item = action[2:]
                if stack and stack[-1] == item:
                    stack.pop()
                else:
                    return []
            else:
                return []
        tmp_ls= random.choice(sorted(list(sequences)))
        if self.is_pruning_r2:
            if len(tmp_ls.split(";"))<=r2_threshold:
                return tmp_ls
            else:
                return []
        return tmp_ls
