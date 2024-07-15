from abc import ABC, abstractmethod
from datasets import Dataset, Value

import os
import copy
import json
import numpy as np
from tqdm import tqdm

import torch

from utils import NpEncoder
import re


class Preprocessor(ABC):
    '''
    Description
    -----------
    Abstract class based on which built-in and custom preprocessors are
    prepared.

    Attributes
    ----------
    checkpoint_dir : str
    max_seq_len : int
    batch_size : int
    length_threshold : int

    Methods
    -------
    process_data():
        Given a loaded dataset from HF or local path, output the preprocessed
        dataset.
    '''
    def __init__(self, checkpoint_dir, max_seq_len, batch_size, mode):
        self.checkpoint_dir = checkpoint_dir
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.mode = mode

    @abstractmethod
    def process_data(self, data):
        '''
        Description
        -----------

        Given a loaded dataset from HF or local path, output the preprocessed
        dataset.

        Parameters
        ----------
        data : ``Dataset``, A dataset object from HF with at least 'label' and
               'text' columns.

        '''
        pass


class Preprocessor_for_RNN(Preprocessor):
    def __init__(self, embed_dir_processed,
                 vocab_size=99, embed_type='glove', embed_size=300, 
                 asset_dir=None, dataset_name=None, is_pruning=False,
                 is_android=False, java_dataset_type=0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dir_processed = embed_dir_processed        
        self.embed_type = embed_type
        self.embed_size = embed_size
        self.asset_dir = asset_dir
        self.dataset_name = dataset_name
        self.is_android = is_android
        self.java_dataset_type = java_dataset_type

        self.tokenizer = self._custom_tokenizer #get_tokenizer('spacy', language='en_core_web_sm')

        self.PAD = '<PAD>'
        self.SOS = '<SOS>'
        self.EOS = '<EOS>'
        self.UNK = '<UNK>'
        self.prune_path_dict={}
        self.adj_matrix = None #self._get_adj_matrix()
        self.unique_paths_matrix = None #self._get_unique_paths_matrix()
        self.is_pruning = is_pruning

        if self.is_android:
            with open(os.path.join(self.asset_dir, self.dataset_name, "v"), "r") as f:
                self.vocab_size = int(f.readline().strip()) + 1
        else:
            self.vocab_size = vocab_size
        

        if self.embed_type not in ['none']:
            self.vocab, self.embeds, self.word2idx, self.idx2word =\
                    self._make_vocab_and_embeds_files()
        else:
            self.vocab, self.embeds, self.word2idx, self.idx2word = (None,)*4
    
    def _custom_tokenizer(self, text):
        word_list = text.split(";")
        return word_list


    def _make_vocab_and_embeds_files(self):
        '''
        Returns:
            vocab (np.ndarray): 1D array of strings, untrimmed vocabulary
            embeds (np.ndarray): 2D array, of untrimmed vocabulary size X
                                 self.embed_size
        '''
        vocab_file = os.path.join(
            self.embed_dir_processed,
            f'vocab_type{self.embed_type}_d{self.embed_size}_np.npy')
        embeds_file = os.path.join(
            self.embed_dir_processed,
            f'embeds_type{self.embed_type}_d{self.embed_size}_np.npy')

        try:
            with open(vocab_file, 'rb') as v_f:
                vocab = np.load(v_f)
            with open(embeds_file, 'rb') as e_f:
                embeds = np.load(e_f)
            print("Loaded vocabulary and embedding files...")
        except FileNotFoundError:
            print("Preparing vocabulary and embedding files...")
            if self.embed_type.lower() == 'glove':
                vocab, embeds = self.make_vocab_and_embeds_glove(
                    vocab_file, embeds_file)
            elif self.embed_type.lower() in ['word2vec', 'w2v']:
                vocab, embeds = self.make_vocab_and_embeds_w2v(
                    vocab_file, embeds_file)
            else:
                raise Exception(
                    "'embed_type' can only be 'glove', 'word2vec', or 'none'.")

        indexes = np.arange(vocab.size)
        word2idx = {}
        idx2word = {}
        for idx, word in zip(indexes, vocab):
            word2idx[word] = idx
            idx2word[idx] = word

        # Additionally adding <UNK> (will be reindexed later)
        word2idx[self.UNK] = len(indexes) + 1
        idx2word[len(indexes) + 1] = self.UNK

        return vocab, embeds, word2idx, idx2word

    def process_data(self, data, train_split=True, rewriting=False,
                     last_checkpoint_path=None, first_shard=True):
        if self.embed_type not in ['none']:
            data = self.process_data_embeds(
                data, train_split=train_split, rewriting=rewriting,
                last_checkpoint_path=last_checkpoint_path,
                first_shard=first_shard)
        else:
            data = self.process_data_no_embeds(
                data, train_split=train_split, rewriting=rewriting,
                last_checkpoint_path=last_checkpoint_path,
                first_shard=first_shard)
        return data

    def process_data_embeds(self, data, train_split=True, rewriting=False,
                            last_checkpoint_path=None, first_shard=True):
        '''
        Procedure:
            1. Tokenize and vectorize raw data
            2. Get the most frequent tokens
            3. Recreate the vocabulary, embeddings and word2idx/idx2word
               dictionaries based on only the most frequent tokens (or based
               on existing saved files when rewriting.)
            4. Reindex the data and pad
        Additionally, if prepending labels:
            Get a set of all labels in the training set as strings.
            Prepend 'LABEL_' to each string in case it is a duplicate
            vocabulary item.
            Create corresponding indexes for each string label, starting from
            the last vocabulary index (self.vocab_size + 4).
            Add both strings and indexes to self.word2idx and self.idx2word
            ...
        '''
        idx_labels = None

        data = self._tokenize_and_vectorize(data)

        if first_shard and train_split:
            if rewriting:
                try:
                    old_idx2word = copy.deepcopy(self.idx2word)
                    self.vocab, self.embeds, self.word2idx, self.idx2word =\
                        self._load_existing_compact_embeds(last_checkpoint_path)
                except:
                    print("Could not load existing word2idx and idx2word "
                          "dictionaries, rebuilding based on specified dataset. "
                          "If pre-training and rewriting on two different "
                          "datasets, MAKE SURE the vocabularies are the same "
                          "for both.")
                    top_idxs = self._get_frequency(data)
                    old_idx2word = copy.deepcopy(self.idx2word)
                    self.vocab, self.embeds, self.word2idx, self.idx2word =\
                        self._prepare_compact_embeds(top_idxs,
                                                     idx_labels=idx_labels)
            else:
                top_idxs = self._get_frequency(data)
                old_idx2word = copy.deepcopy(self.idx2word)
                self.vocab, self.embeds, self.word2idx, self.idx2word =\
                    self._prepare_compact_embeds(top_idxs,
                                                 idx_labels=idx_labels)
        else:
            old_idx2word = None

        len_labels = 0

        data = self._reindex_data_and_pad(data, self.word2idx,
                                          old_idx2word=old_idx2word)

        if first_shard and train_split:
            self.vocab_size = self.vocab_size + 4 + len_labels

        return data

    def _tokenize_and_vectorize(self, dataset):
        data = []
        for doc_dict in tqdm(dataset):
            tokenized = self.tokenizer(doc_dict['text'].strip())
            tensor = torch.tensor(
                [self.word2idx[token]
                 if token in self.word2idx.keys()
                 else self.word2idx[self.UNK]
                 for token in tokenized][:(self.max_seq_len-2)],
                dtype=torch.long)  # -2 for SOS+EOS tokens
            length = tensor.size()[0]
            data.append((tensor, length, doc_dict['label']))

        return data

    def _get_frequency(self, train_data):
        relevant = torch.cat([val[0] for val in train_data])
        # Ignore UNK token in counts
        relevant = relevant[relevant != self.word2idx[self.UNK]]
        counts = torch.bincount(relevant)
        top_counts, top_indexes = torch.topk(counts, self.vocab_size)
        return top_indexes

    def _prepare_compact_embeds(self, top_indexes, idx_labels=None):
        reindexes = np.arange(self.vocab_size+4)
        new_vocab = self.vocab[top_indexes]
        new_embeds = self.embeds[top_indexes, :]

        new_vocab = np.insert(new_vocab, 0, self.PAD)
        new_vocab = np.insert(new_vocab, 1, self.UNK)
        new_vocab = np.insert(new_vocab, 2, self.SOS)
        new_vocab = np.insert(new_vocab, 3, self.EOS)

        # Pad token is all 0s
        pad_emb_np = np.zeros((1, new_embeds.shape[1]))
        # UNK token is mean of all other embeds
        unk_emb_np = np.mean(new_embeds, axis=0, keepdims=True)
        # SOS token is a random vector (standard normal)
        sos_emb_np = np.random.normal(size=pad_emb_np.shape)
        # EOS token is a random vector (standard normal)
        eos_emb_np = np.random.normal(size=pad_emb_np.shape)

        new_embeds = np.vstack((pad_emb_np, unk_emb_np, sos_emb_np,
                                eos_emb_np, new_embeds))
        new_word2idx = {}
        new_idx2word = {}
        for idx, word in zip(reindexes, new_vocab):
            new_word2idx[word] = idx
            new_idx2word[idx] = word

        # Save vocabulary, embeddings, word2idx and idx2word in checkpoint
        # directory
        np.save(os.path.join(self.checkpoint_dir, 'vocab.npy'), new_vocab)
        np.save(os.path.join(self.checkpoint_dir, 'embeds.npy'), new_embeds)
        with open(os.path.join(self.checkpoint_dir, 'word2idx.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(new_word2idx, f, ensure_ascii=False, indent=4,
                      cls=NpEncoder)

        return new_vocab, new_embeds, new_word2idx, new_idx2word

    def _load_existing_compact_embeds(self, last_checkpoint_path):
        '''
        For rewriting mode, load the pre-trained vocabulary, embeddings,
        word2idx and idx2word dictionaries.
        '''
        checkpoint_dir = os.path.abspath(
            os.path.join(last_checkpoint_path, os.pardir))
        new_vocab = np.load(os.path.join(checkpoint_dir, 'vocab.npy'))
        new_embeds = np.load(os.path.join(checkpoint_dir, 'embeds.npy'))
        with open(os.path.join(checkpoint_dir, 'word2idx.json'), 'r',
                  encoding='utf-8') as f:
            new_word2idx = json.load(f)
        new_idx2word = {idx: word for word, idx in new_word2idx.items()}

        return new_vocab, new_embeds, new_word2idx, new_idx2word

    def _reindex_data_and_pad(self, data, compact_word2idx, old_idx2word=None):
        '''
        Carries out three tasks:
        1. Converts indexes of untrimmed vocabulary to indexes of trimmed
        vocabulary.
        2. Adds special tokens to existing tensor (SOS, EOS and PAD).
        '''
        reindexed = []
        for data_point in tqdm(data):
            if old_idx2word is not None:
                old_indexes = data_point[0]
                words = [old_idx2word[idx.item()] for idx in old_indexes]
                new_indexes = torch.tensor(
                        [compact_word2idx[word]
                         if word in compact_word2idx
                         else compact_word2idx[self.UNK] for word in words],
                        dtype=torch.long)
            else:
                # If subsequent shard
                new_indexes = data_point[0]

            lab = data_point[2]

            if self.is_pruning:
                new_indexes=torch.tensor(self._prune_path(new_indexes),dtype=torch.long)

            new_indexes = torch.cat(
                    (torch.tensor([compact_word2idx[self.SOS]]),
                     new_indexes,
                     torch.tensor([compact_word2idx[self.EOS]])), dim=0)
            new_length = data_point[1] + 2

            while new_indexes.shape[0] < self.max_seq_len:
                new_indexes = torch.cat(
                        (new_indexes,
                         torch.tensor([compact_word2idx[self.PAD]])))

            reindexed.append((new_indexes, new_length, lab))

        return reindexed

    def process_data_no_embeds(self, data, train_split=True, rewriting=False,
                               last_checkpoint_path=None, first_shard=True):
        # Only prepare the vocabulary based on the first shard
        # (dataset should be large enough, e.g. Wikipedia)
        if first_shard and train_split:
            data = self._process_data_no_embeds_first_shard(
                data, rewriting=rewriting,
                last_checkpoint_path=last_checkpoint_path)
        else:
            data = self._process_data_no_embeds_subsequent_shard(
                data, rewriting=rewriting,
                last_checkpoint_path=last_checkpoint_path)
        return data

    def _create_adj_matrix(self,train_data,valid_data):
        '''
        Returns the adjacency matrix for the vocabulary.
        '''
        if self.is_android:
            if not os.path.exists(os.path.join(self.asset_dir,self.dataset_name, "call_matrix.npy")):
                call_pairs = set()
                self.adj_matrix = np.zeros((self.vocab_size, self.vocab_size),dtype=np.float32)
                for data_point in sorted(list(set(train_data) | set(valid_data)),key=lambda x: x[1]):
                    result = set([(data_point[0][i].item(), data_point[0][i+1].item()) for i in range(data_point[1]-1)])
                    call_pairs = call_pairs | result
                for pair in call_pairs:
                    self.adj_matrix[pair[0], pair[1]] = 1
                np.save(os.path.join(self.asset_dir,self.dataset_name, "call_matrix.npy"),self.adj_matrix)
            else:
                self.adj_matrix = np.load(os.path.join(self.asset_dir,self.dataset_name, "call_matrix.npy"))
        else:
            if not os.path.exists(os.path.join(self.asset_dir,self.dataset_name, "call_pairs.json")):
                call_pairs = set()
                for data_point in sorted(list(set(train_data) | set(valid_data)),key=lambda x: x[1]):
                    result = set([(data_point[0][i].item(), data_point[0][i+1].item()) for i in range(data_point[1]-1)])
                    call_pairs = call_pairs | result
                with open(os.path.join(self.asset_dir,self.dataset_name, "call_pairs.json"), "w") as f:
                    json.dump(sorted(list(call_pairs)), f)
            else:
                with open(os.path.join(self.asset_dir,self.dataset_name, "call_pairs.json"), "r") as f:
                    call_pairs = json.load(f)
                    call_pairs = set([(x[0], x[1]) for x in call_pairs])
            adj_matrix = torch.zeros((self.vocab_size, self.vocab_size))
            for pair in call_pairs:
                adj_matrix[pair[0], pair[1]] = 1
            self.adj_matrix = np.array(adj_matrix)

        if not os.path.exists(os.path.join(self.asset_dir,self.dataset_name, "unique_paths_matrix.npy")):
            self.unique_paths_matrix = self.calculate_transitive_closure()
            np.save(os.path.join(self.asset_dir,self.dataset_name, "unique_paths_matrix.npy"),self.unique_paths_matrix)
        else:
            self.unique_paths_matrix = np.load(os.path.join(self.asset_dir,self.dataset_name, "unique_paths_matrix.npy"))
        return

    def _get_adj_matrix(self):
        '''
        Returns the adjacency matrix for the vocabulary.
        '''
        if self.is_android:
            with open(os.path.join(self.asset_dir, self.dataset_name, "callpairs"), "r") as f:
                call_pairs=set()
                for line in f:
                    node1, node2 = line.strip().split(',')
                    call_pairs.add((self.word2idx[node1], self.word2idx[node2]))      
        else:
            with open(os.path.join(self.asset_dir,self.dataset_name, "call_pairs.json"), "r") as f:
                    call_pairs = json.load(f)
                    call_pairs = set([(x[0], x[1]) for x in call_pairs])

        adj_matrix = torch.zeros((self.vocab_size, self.vocab_size))
        for pair in call_pairs:
            adj_matrix[pair[0], pair[1]] = 1
        feature_matrix = np.array(adj_matrix)
        return feature_matrix
    
    def _get_unique_paths_matrix(self):
        n = len(self.adj_matrix)
        unique_paths_matrix = [[0] * n for _ in range(n)]
        
        for i in range(4,n):
            for j in range(4,n):
                if i != j:
                    paths = self._find_paths(i, j)
                    if len(paths) == 1:
                        if (i,j) not in self.prune_path_dict:
                            self.prune_path_dict[(i,j)] = paths[0]
                        unique_paths_matrix[i][j] = 1

        return unique_paths_matrix

    def calculate_transitive_closure(self):
        """
        Calculate the transitive closure of the adjacency matrix.
        """
        num_vertices = self.adj_matrix.shape[0]
        adj_matrix = torch.tensor(self.adj_matrix).to('cuda' if torch.cuda.is_available() else 'cpu')

        adjacency_power = torch.eye(num_vertices,dtype=adj_matrix.dtype).to(adj_matrix.device)
        sum_adjacency_power = torch.zeros((num_vertices, num_vertices)).to(adj_matrix.device)

        for i in range(1, num_vertices + 1):
            adjacency_power = torch.mm(adjacency_power, adj_matrix)
            sum_adjacency_power += adjacency_power

        return (sum_adjacency_power==1).cpu().numpy().astype(int)

    def _find_paths(self, start, end):
        paths = []
        self._dfs(start, end, [], paths)
        return paths

    def _dfs(self, curr, end, path, paths):
        if len(paths) > 1:
            return
        
        path.append(curr)
        if curr == end:
            paths.append(list(path))
        else:
            neighbors = np.where(self.adj_matrix[curr] == 1)[0]
            for neighbor in neighbors:#range(len(self.adj_matrix[curr])):
                if neighbor not in path:
                    self._dfs(neighbor, end, path, paths)
        path.pop()
    
    def _prune_path(self, path):
        if isinstance(path, torch.Tensor):
            pruned_path = path.tolist()
        else:
            pruned_path = list(path)

        i = 0

        while i < len(pruned_path) - 2:
            start = pruned_path[i]
            end = pruned_path[i+2]

            if self.unique_paths_matrix[start][end] == 1:
                pruned_path = pruned_path[:i+1] + pruned_path[i+2:]
            else:
                i += 1

        return pruned_path

    def _restore_path(self, pruned_path):
        if isinstance(pruned_path, torch.Tensor):
            restored_path = pruned_path.tolist()
        else:
            restored_path = list(pruned_path)
        
        i = 0

        while i < len(restored_path) - 1:
            start = restored_path[i]
            end = restored_path[i+1]

            if self.unique_paths_matrix[start][end] == 1:
                if (start,end) in self.prune_path_dict:
                    subpath = self.prune_path_dict[(start,end)]
                else:
                    subpath = self._find_paths(start, end)[0]
                restored_path[i:i + 1] = subpath[:-1]
                i += len(subpath) - 1
            else:
                i += 1

        return restored_path

    def _process_data_no_embeds_first_shard(
            self, data, last_checkpoint_path=None, rewriting=False):
        def encode(examples):
            examples['text'] = [self.tokenizer(doc.strip()) for doc in examples['text']]
            return examples

        # Multiprocessing for larger datasets
        threshold = 50000
        if len(data) > threshold:
            proc_num = None#os.cpu_count()
        else:
            proc_num = None

        data = data.map(encode, batched=True, num_proc=proc_num)
        labels = data['label']

        if self.is_android:
            self.idx2word = {i:str(i) for i in range(self.vocab_size)}
        else:
            if os.path.exists(os.path.join(self.asset_dir,self.dataset_name, 'large_idx2word.json')):
                try:
                    checkpoint_dir = self.checkpoint_dir
                    #os.path.abspath(os.path.join(last_checkpoint_path, os.pardir))
                    with open(os.path.join(self.asset_dir,self.dataset_name,
                            'large_idx2word.json'), 'r',
                            encoding='utf-8') as f:
                        self.idx2word = json.load(f)
                    self.idx2word = {int(idx): token
                                    for idx, token in self.idx2word.items()}
                except FileNotFoundError:
                    print("Could not load existing FULL word2idx and idx2word "
                        "dictionaries, rebuilding based on specified dataset. "
                        "If pre-training and rewriting on two different "
                        "datasets, MAKE SURE the vocabularies are the same "
                        "for both.")
                    idx2word = []
                    for idx, doc in tqdm(enumerate(data)):
                        idx2word += doc['text']
                        if idx % 5000 == 0:
                            idx2word = list(set(idx2word))
                    idx2word = list(set(idx2word))
                    self.idx2word = {idx: token
                                    for idx, token in enumerate(idx2word)}
            else:
                idx2word = []
                for idx, doc in tqdm(enumerate(data)):
                    idx2word += doc['text']
                idx2word = sorted(list(set(idx2word)))

                self.idx2word = {idx: token for idx, token in enumerate(idx2word)}

                with open(os.path.join(self.asset_dir,self.dataset_name,
                        'large_idx2word.json'), 'w',
                        encoding='utf-8') as f:
                    json.dump(self.idx2word, f, ensure_ascii=False, indent=4,
                            cls=NpEncoder)

        # Additionally adding <UNK> (will be reindexed later)
        self.idx2word[len(self.idx2word) + 1] = self.UNK
        self.word2idx = {token: idx for idx, token in self.idx2word.items()}
        data = data.map(self._encode, batched=True, num_proc=proc_num)
        data = [(torch.tensor(indexes), length, label)
                  for indexes, length, label in zip(
                      data['encoded'], data['length'], data['label'])]

        if len(self.idx2word) < self.vocab_size:
            self.vocab_size = len(self.idx2word) - 1

        if os.path.exists(os.path.join(self.asset_dir,self.dataset_name, 'idx2word.json')):
            # Loading the small idx2word
            try:
                old_idx2word = copy.deepcopy(self.idx2word)
                with open(os.path.join(self.asset_dir,self.dataset_name, 'idx2word.json'), 'r',
                          encoding='utf-8') as f:
                    self.idx2word = json.load(f)
                self.idx2word = {int(idx): token for idx, token in self.idx2word.items()}
                self.word2idx = {token: idx for idx, token in self.idx2word.items()}
            except FileNotFoundError:
                print("Could not load existing TRIMMED word2idx and idx2word "
                      "dictionaries, rebuilding based on specified dataset. "
                      "If pre-training and rewriting on two different "
                      "datasets, MAKE SURE the vocabularies are the same "
                      "for both.")
                top_indexes = self._get_frequency(data)
                old_idx2word = copy.deepcopy(self.idx2word)
                reindexes = np.arange(self.vocab_size+4)
                vocab = np.vectorize(self.idx2word.get)(top_indexes)
                vocab = np.insert(vocab, 0, self.PAD)
                vocab = np.insert(vocab, 1, self.UNK)
                vocab = np.insert(vocab, 2, self.SOS)
                vocab = np.insert(vocab, 3, self.EOS)
                new_word2idx = {}
                new_idx2word = {}
                for idx, word in zip(reindexes, vocab):
                    new_word2idx[word] = idx
                    new_idx2word[idx] = word
                self.idx2word = new_idx2word
                self.word2idx = new_word2idx

                with open(os.path.join(self.asset_dir,self.dataset_name, 'idx2word.json'), 'w',
                          encoding='utf-8') as f:
                    json.dump(self.idx2word, f, ensure_ascii=False, indent=4,
                              cls=NpEncoder)
        else:
            old_idx2word = copy.deepcopy(self.idx2word)
            reindexes = np.arange(self.vocab_size+4)
            if self.is_android:
                top_indexes=[i for i in range(len(self.idx2word))]
                vocab = np.vectorize(self.idx2word.get)(top_indexes)
                if int(str(vocab.dtype)[2:])<5:
                    vocab = vocab.astype('<U5')
            else:
                top_indexes = self._get_frequency(data)
                vocab = np.vectorize(self.idx2word.get)(top_indexes)
            
            vocab = np.insert(vocab, 0, self.PAD)
            vocab = np.insert(vocab, 1, self.UNK)
            vocab = np.insert(vocab, 2, self.SOS)
            vocab = np.insert(vocab, 3, self.EOS)
            new_word2idx = {}
            new_idx2word = {}
            for idx, word in zip(reindexes, vocab):
                new_word2idx[word] = int(idx)
                new_idx2word[int(idx)] = word
            self.idx2word = new_idx2word
            self.word2idx = new_word2idx

            with open(os.path.join(self.asset_dir,self.dataset_name, 'idx2word.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(self.idx2word, f, ensure_ascii=False, indent=4,
                          cls=NpEncoder)

        self.vocab_size = len(self.idx2word)
        if self.is_pruning:# and not self.is_android:
            self._create_adj_matrix(None, None)
            '''self.adj_matrix = self._get_adj_matrix()
            self.unique_paths_matrix = self._get_unique_paths_matrix()'''
        
        data = self._reindex_data_and_pad(data, self.word2idx,
                                          old_idx2word=old_idx2word)
        return data

    def _process_data_no_embeds_subsequent_shard(
            self, data, last_checkpoint_path=None, rewriting=False):
        def encode(examples):
            examples['text'] = [self.tokenizer(doc.strip())
                                for doc in examples['text']]
            return examples

        # Multiprocessing for larger datasets
        threshold = 50000
        if len(data) > threshold:
            proc_num = None #os.cpu_count()
        else:
            proc_num = None
        data = data.map(encode, batched=True, num_proc=proc_num)
        data = data.map(self._encode, batched=True)
        data = [(torch.tensor(indexes), length, label)
                for indexes, length, label in zip(
                    data['encoded'], data['length'], data['label'])]
        # Only padding and adding sos/eos tokens in this case
        data = self._reindex_data_and_pad(data, self.word2idx)

        return data

    def _encode(self, examples):
        encoded = [[self.word2idx[tok] if tok in self.word2idx else self.word2idx[self.UNK] for tok in doc] for doc in examples['text']]
        examples['encoded'] = [torch.tensor(enc_doc[:(self.max_seq_len-2)]) for enc_doc in encoded]
        examples['length'] = [doc.size()[0] for doc in examples['encoded']]
        return examples

