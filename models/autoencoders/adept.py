from models.autoencoders.general_classes import Autoencoder_RNN, GeneralModelConfig
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random
import numpy as np
from queue import PriorityQueue
import operator

class ADePTModelConfig(GeneralModelConfig):
    def __init__(self, pretrained_embeddings=None, vocab_size=20000,
                 no_clipping=False,dp_module='clip_norm', **kwargs):
        super().__init__(**kwargs)
        self.pretrained_embeddings = pretrained_embeddings
        self.vocab_size = vocab_size
        self.no_clipping = no_clipping
        self.dp_module = dp_module
        self.use_call_matrix = kwargs['use_call_matrix']
        self.where_matrix = kwargs['where_matrix']
        self.use_cfg = kwargs['use_cfg']
        


class ADePT(Autoencoder_RNN):
    def __init__(self, config, feature_matrix=None, adj_matrix=None):
        super().__init__()
        self.config = config
        self.encoder = self.get_encoder()
        self.feature_matrix = feature_matrix
        self.adj_matrix = torch.from_numpy(adj_matrix).to(self.config.device)
        if self.config.use_call_matrix:
            self.feature_matrix=torch.from_numpy(feature_matrix).to(self.config.device)
            self.decoder = self.get_decoder(use_call_matrix=True)
        else:
            self.decoder = self.get_decoder()

    def get_encoder(self):
        encoder = Encoder(
            self.config.pretrained_embeddings,
            self.config.embed_size, self.config.hidden_size,
            self.config.enc_out_size, self.config.vocab_size,
            batch_size=self.config.batch_size, num_layers=1, dropout=0.0,
            pad_idx=self.config.pad_idx)
        return encoder

    def get_decoder(self, use_call_matrix=False):
        decoder = Decoder(
            self.config.pretrained_embeddings, self.config.embed_size,
            self.config.hidden_size, self.config.enc_out_size,
            self.config.vocab_size, device=self.config.device,
            batch_size=self.config.batch_size, num_layers=1, dropout=0.0,
            pad_idx=self.config.pad_idx,
            use_call_matrix=use_call_matrix, feature_matrix=self.feature_matrix, 
            where_matrix=self.config.where_matrix)
        return decoder

    def forward(self, **inputs):
        '''
        Input:
            inputs: batch_size X seq_len
            lengths: batch_size
        Return:
            z: batch_size X seq_len X vocab_size+4
            enc_embed: batch_size X seq_len X embed_size
        '''
        input_ids = inputs['input_ids']
        lengths = inputs['lengths']
        teacher_forcing_ratio = inputs['teacher_forcing_ratio']
        encoder_hidden = self.encoder.initHidden(batch_size=input_ids.shape[0])
        encoder_hidden = (encoder_hidden[0].to(self.config.device),
                          encoder_hidden[1].to(self.config.device))

        encoder_out, encoder_hidden = self.encoder(input_ids, lengths,
                                                   encoder_hidden)

        ## Privacy module
        context_vec = torch.cat((encoder_hidden[0], encoder_hidden[1]), dim=2)
        
        if self.config.private:
            context_vec = self.privatize(context_vec)
        else:
            if not self.config.no_clipping:
                # ADePT model clips without privacy to encourage model
                # representations to be within a radius of C
                context_vec = self.clip(context_vec)
        encoder_hidden = (
            context_vec[:, :, :self.config.hidden_size].contiguous(),
            context_vec[:, :, self.config.hidden_size:].contiguous()
            )

        target = deepcopy(input_ids)
        z = torch.zeros(input_ids.shape[0], self.config.max_seq_len,
                        self.config.vocab_size)
        z = z.to(self.config.device)
        prev_x = target[:, 0]
        #prev_x = prev_x.unsqueeze(0)
        prev_x = prev_x.unsqueeze(1)

        decoder_hidden = encoder_hidden

        teacher_force = random.random() < teacher_forcing_ratio
        if teacher_force:
            for i in range(1, self.config.max_seq_len):
                z_i, decoder_hidden = self.decoder(prev_x, decoder_hidden)
                #prev_x = target[:, i].unsqueeze(dim=0)
                prev_x = target[:, i].unsqueeze(dim=1)
                z[:, i] = z_i
        else:
            for i in range(1, self.config.max_seq_len):
                z_i, decoder_hidden = self.decoder(prev_x, decoder_hidden)
                top_idxs = z_i.argmax(1)
                prev_x = top_idxs.unsqueeze(dim=1)#0
                z[:, i] = z_i
        z = z[:, 1:, :].reshape(-1, z.shape[2]) #z.transpose(1, 0)[:, 1:, :].reshape(-1, z.shape[2])

        return z

    def privatize(self, context_vec):
        if self.config.dp_module == 'clip_norm':
            clipped = self.clip(context_vec)
            noisified = self.noisify(clipped, context_vec)
        elif self.config.dp_module == 'clip_value':
            sigma = 0.2
            mean = 0.00
            num_sigmas = 1
            left_clip = mean - (sigma * num_sigmas)
            right_clip = mean + (sigma * num_sigmas)
            clipped_tensor = torch.clamp(context_vec, left_clip, right_clip)
            if True:
                k = context_vec.shape[-1]
                if self.config.dp_mechanism == 'laplace':
                    sensitivity = 2 * sigma * num_sigmas * k
                    laplace = torch.distributions.laplace.Laplace(0, sensitivity / self.config.epsilon)
                    noise = laplace.sample(sample_shape=torch.Size((clipped_tensor.shape[0], clipped_tensor.shape[1])))
                elif self.config.dp_mechanism == 'gaussian':
                    sensitivity = 2 * sigma * num_sigmas * np.sqrt(k)
                    scale = np.sqrt((sensitivity**2 / self.config.epsilon**2) * 2 * np.log(1.25 / self.config.delta))
                    gauss = torch.distributions.normal.Normal(0, scale)
                    noise = gauss.sample(sample_shape=torch.Size((clipped_tensor.shape[0], clipped_tensor.shape[1], clipped_tensor.shape[2])))
                else:
                    raise Exception(f"No DP mechanism available called '{self.config.dp_mechanism}'.")
                noise = noise.to(self.config.device)
                noisified = clipped_tensor + noise
        return noisified

    def clip(self, context_vec):
        norm = torch.linalg.norm(context_vec, axis=2, ord=self.config.norm_ord)
        ones = torch.ones(norm.shape[1]).to(self.config.device)
        min_val = torch.minimum(ones, self.config.clipping_constant / norm)
        clipped = min_val.unsqueeze(-1) * context_vec
        return clipped

    def get_sensitivity_for_clip_by_norm(self, clipped_tensor):
        if self.config.norm_ord == 1 and self.config.dp_mechanism == 'laplace':
            sensitivity = torch.tensor(2 * self.config.clipping_constant)
        elif self.config.norm_ord == 2 and\
                self.config.dp_mechanism == 'laplace':
            sensitivity = 2 * self.config.clipping_constant * torch.sqrt(
                torch.tensor(clipped_tensor.shape[2]))
        elif self.config.norm_ord == 2 and\
                self.config.dp_mechanism == 'gaussian':
            sensitivity = torch.tensor(2 * self.config.clipping_constant)
        else:
            raise Exception("Sensitivity calculation for clipping by norm only implemented for Laplace mechanism with L1/L2 norm clipping, or Gaussian mechanism with L2 norm clipping.")
        return sensitivity

    def noisify(self, clipped_tensor, context_vec):
        sensitivity = self.get_sensitivity_for_clip_by_norm(clipped_tensor)
        if self.config.dp_mechanism == 'laplace':
            laplace = torch.distributions.Laplace(
                0, sensitivity / self.config.epsilon)
            noise = laplace.sample(
                sample_shape=torch.Size((context_vec.shape[1],
                                         context_vec.shape[2]))).unsqueeze(0)
        elif self.config.dp_mechanism == 'gaussian':
            scale = torch.sqrt(
                (sensitivity**2 / self.config.epsilon**2) * 2 * torch.log(torch.tensor(1.25 / self.config.delta)))
            gauss = torch.distributions.normal.Normal(0, scale)
            noise = gauss.sample(
                sample_shape=torch.Size((clipped_tensor.shape[1],
                                         clipped_tensor.shape[2])))
        else:
            raise Exception(f"No DP mechanism available called '{self.dp_mechanism}'.")
        noise = noise.to(self.config.device)
        noisified = clipped_tensor + noise
        return noisified
    
    def decode(self, method='beam-search', call_pairs=None, beam_size=3, **inputs):
        input_ids = inputs['input_ids']
        lengths = inputs['lengths']
        teacher_forcing_ratio = inputs['teacher_forcing_ratio']
        encoder_hidden = self.encoder.initHidden()
        encoder_hidden = (encoder_hidden[0].to(self.config.device),
                          encoder_hidden[1].to(self.config.device))

        encoder_out, encoder_hidden = self.encoder(input_ids, lengths,
                                                   encoder_hidden)

        ## Privacy module
        context_vec = torch.cat((encoder_hidden[0], encoder_hidden[1]), dim=2)
        if self.config.private:
            context_vec = self.privatize(context_vec)
        else:
            if not self.config.no_clipping:
                # ADePT model clips without privacy to encourage model
                # representations to be within a radius of C
                context_vec = self.clip(context_vec)
        encoder_hidden = (
            context_vec[:, :, :self.config.hidden_size].contiguous(),
            context_vec[:, :, self.config.hidden_size:].contiguous()
            )

        target = deepcopy(input_ids)
        if beam_size > 1:
            return self.beam_decode(target, encoder_hidden, encoder_out,call_pairs=call_pairs, beam_size=beam_size)
        else:
            return self.greedy_decode(target, encoder_hidden, encoder_out)

    def greedy_decode(self, trg, decoder_hiddens, encoder_outputs):
        SOS_token = 2
        EOS_token = 3

        batch_size, seq_len = trg.size()
        score_batch = torch.ones((batch_size, 1), dtype=torch.float32, device=trg.device)
        decoder_inputs = torch.zeros((batch_size, 1, seq_len), dtype=torch.long, device=trg.device)
        eos_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=trg.device)
        decoder_hidden = (
            decoder_hiddens[0],
            decoder_hiddens[1]
        )

        decoder_input = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=trg.device)

        for t in range(seq_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            if self.config.use_cfg:
                valid_mask = self.adj_matrix[decoder_input[:, -1]].bool()
                valid_mask[:, EOS_token] = True
            else:
                valid_mask = torch.ones_like(decoder_output, dtype=torch.bool, device=trg.device)

            valid_decoder_output = decoder_output.masked_fill(~valid_mask, -float('inf'))
            log_prob, indexes = torch.topk(valid_decoder_output, 1)

            decoder_input = indexes.view(-1, 1)
            decoder_inputs[:, 0, t] = decoder_input.view(-1)

            eos_mask = eos_mask | (decoder_input == EOS_token)
            decoder_inputs[eos_mask.view(-1), 0, t] = EOS_token
            if eos_mask.all():
                break

        return decoder_inputs, score_batch

    def beam_decode(self, target_tensor, decoder_hiddens, encoder_outputs=None,call_pairs=None, beam_size=3):
        '''
        :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        :param decoder_hiddens: input tensor of shape [1, B, H] for start of the decoding
        :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        :return: decoded_batch
        '''
        #target_tensor = target_tensor.permute(1, 0)
        topk = beam_size  # how many sentence do you want to generate
        beam_width = min(beam_size,self.config.vocab_size)
        decoded_batch = []
        score_batch = []
        SOS_token = 2
        EOS_token = 3

        # decoding goes sentence by sentence
        for idx in range(target_tensor.size(0)):  # batch_size
            # z_i, decoder_hidden = self.decoder(prev_x, decoder_hidden)
            if isinstance(decoder_hiddens, tuple):  # LSTM case
                decoder_hidden = (
                    decoder_hiddens[0][:, idx, :].unsqueeze(0), decoder_hiddens[1][:, idx, :].unsqueeze(0))
            else:
                decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)  # [1, B, H]=>[1,H]=>[1,1,H]
            encoder_output = encoder_outputs[:, idx, :].unsqueeze(1)  # [T,B,H]=>[T,H]=>[T,1,H]
            
            # Start with the start of the sentence token
            decoder_input = torch.LongTensor([SOS_token]).to(target_tensor.device)

            # Number of sentence to generate
            endnodes = []
            number_required = min((topk + 1), topk - len(endnodes))

            # starting node -  hidden vector, previous node, word id, logp, length
            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
            nodes = PriorityQueue()

            # start the queue
            nodes.put((-node.eval(), node))
            qsize = 1

            # start beam search
            while True:
                # give up when decoding takes too long
                if qsize > 500: break

                # fetch the best node
                score, n = nodes.get()
                # print('--best node seqs len {} '.format(n.leng))
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == EOS_token and n.prevNode != None:
                    endnodes.append((score, n))
                    # if we reached maximum # of sentences required
                    if len(endnodes) >= number_required:
                        break
                    else:
                        continue

                # decode for one step using decoder
                decoder_output, decoder_hidden = self.decoder(decoder_input.unsqueeze(0), decoder_hidden)#, encoder_outputs)
                log_softmax_output = F.log_softmax(decoder_output, dim=1)

                # PUT HERE REAL BEAM SEARCH OF TOP
                if self.config.use_cfg:# and n.wordid.item() is not SOS_token:
                    valid_nodes = torch.cat(
                        [torch.nonzero(self.adj_matrix[n.wordid.item()]).view(-1),torch.tensor([EOS_token]).to(target_tensor.device)]
                    )
                    valid_log_softmax_output = log_softmax_output[:, valid_nodes]
                    valid_beam_width = min(beam_width, valid_log_softmax_output.shape[1])
                    log_prob, indexes = torch.topk(valid_log_softmax_output, valid_beam_width)
                    indexes = valid_nodes[indexes[0]]
                else:
                    log_prob, indexes = torch.topk(log_softmax_output, beam_width)
                    indexes = indexes[0]
                    valid_beam_width = beam_width
                nextnodes = []

                valid_beam_count=0
                for new_k in range(valid_beam_width):
                    decoded_t = indexes[new_k].view(-1)                   
                    log_p = log_prob[0][new_k].item()
                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, node))
                    valid_beam_count+=1

                # put them into queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    nodes.put((score, nn))
                    # increase qsize
                qsize += len(nextnodes) - 1

            # choose nbest paths, back trace them
            if len(endnodes) == 0:
                endnodes = [nodes.get() for _ in range(topk)]

            utterances = []
            scores = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                beam_score = -score
                scores.append(beam_score)

                utterance = []
                utterance.append(n.wordid)
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(n.wordid)

                utterance = torch.cat(utterance[::-1], dim=0)
                utterances.append(utterance)
            scores = torch.FloatTensor(scores)
            decoded_batch.append(utterances)
            score_batch.append(scores)

        return decoded_batch, score_batch


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward  

    def __lt__(self, other):
        return self.leng < other.leng  

    def __gt__(self, other):
        return self.leng > other.leng


class Encoder(nn.Module):
    def __init__(self, embeds, embed_size, hidden_size, enc_out_size,
                 vocab_size, num_layers=2, batch_size=32, dropout=0.,
                 pad_idx=0,is_dp_lstm=False):
        super(Encoder, self).__init__()
        if embeds is not None:
            self.pretrained_embeds = True
            self.embeds = nn.Embedding.from_pretrained(
                    torch.from_numpy(embeds).float(),
                    padding_idx=pad_idx)
        else:
            self.pretrained_embeds = False
            self.embeds = nn.Embedding(vocab_size, embed_size,
                                       padding_idx=pad_idx)
        
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=False,
                            dropout=dropout)

        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def forward(self, x, lengths, hidden_tup):
        """
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        if self.pretrained_embeds:
            with torch.no_grad():
                x = self.embeds(x)
        else:
            x = self.embeds(x)

        packed = pack_padded_sequence(x, lengths, batch_first=True,
                                      enforce_sorted=False)
        output, (hidden, cell) = self.rnn(packed, hidden_tup)
        output, _ = pad_packed_sequence(output)
        return output, (hidden, cell)

    def initHidden(self,batch_size=None):
        if batch_size is None:
            # 1 X batch_size X hidden_size
            hidden = torch.zeros(1, self.batch_size, self.hidden_size)
            cell = torch.zeros(1, self.batch_size, self.hidden_size)
        else:
            hidden = torch.zeros(1, batch_size, self.hidden_size)
            cell = torch.zeros(1, batch_size, self.hidden_size)
        return (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, embeds, embed_size, hidden_size, enc_out_size,
                 vocab_size, device='cpu', batch_size=32, num_layers=2,
                 dropout=0.0, pad_idx=0,use_call_matrix=False,
                 feature_matrix=None, where_matrix=1):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.device = device
        self.use_call_matrix = use_call_matrix
        self.feature_matrix = feature_matrix
        self.where_matrix = where_matrix

        if embeds is not None:
            self.pretrained_embeds = True
            self.embeds = nn.Embedding.from_pretrained(
                    torch.from_numpy(embeds).float(),
                    padding_idx=pad_idx)
        else:
            self.pretrained_embeds = False
            self.embeds = nn.Embedding(vocab_size, embed_size,
                                       padding_idx=pad_idx)

        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=False,
                            dropout=dropout)#batch_first=False
        if not use_call_matrix:
            self.generator = nn.Linear(hidden_size, vocab_size)
        else:
            self.generator = nn.Linear(hidden_size+self.feature_matrix.shape[1], vocab_size)

    def forward(self, prev_x, hidden_tup):

        # update rnn hidden state
        if self.pretrained_embeds:
            with torch.no_grad():
                prev_embed = self.embeds(prev_x)
        else:
            prev_embed = F.relu(self.embeds(prev_x))
        output, (hidden, cell) = self.rnn(prev_embed, hidden_tup)
        if self.use_call_matrix and self.where_matrix == 1:
            output = torch.cat((output,self.feature_matrix[prev_x]),dim=2)
        z_i = self.generator(output[:,0,:])

        return z_i, (hidden, cell)




