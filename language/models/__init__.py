from .encoder import Encoder
from .decoder import Decoder
from .seq2seq import Seq2seq

def get_vocab(opt):
    if opt.dataset == 'clevr':
        vocab_json = opt.clevr_vocab_path
    else:
        raise ValueError('Invalid dataset')
    vocab = load_vocab(vocab_json)
    return vocab


def create_seq2seq_net(input_vocab_size, output_vocab_size, hidden_size, 
                       word_vec_dim, n_layers, bidirectional, variable_lengths, 
                       use_attention, encoder_max_len, decoder_max_len, start_id, 
                       end_id, word2vec_path=None, fix_embedding=False):
    word2vec = None
    if word2vec_path is not None:
        word2vec = load_embedding(word2vec_path)

    encoder = Encoder(input_vocab_size, encoder_max_len, 
                      word_vec_dim, hidden_size, n_layers,
                      bidirectional=bidirectional, variable_lengths=variable_lengths,
                      word2vec=word2vec, fix_embedding=fix_embedding)
    decoder = Decoder(output_vocab_size, decoder_max_len,
                      word_vec_dim, hidden_size, n_layers, start_id, end_id,
                      bidirectional=bidirectional, use_attention=use_attention)

    return Seq2seq(encoder, decoder)

import os
import json
import numpy as np
import torch


def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)


def invert_dict(d):
  return {v: k for k, v in d.items()}
  

def load_vocab(path):
    with open(path, 'r') as f:
        vocab = json.load(f)
        vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
        vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
        vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
    # Sanity check: make sure <NULL>, <START>, and <END> are consistent
    assert vocab['question_token_to_idx']['<NULL>'] == 0
    assert vocab['question_token_to_idx']['<START>'] == 1
    assert vocab['question_token_to_idx']['<END>'] == 2
    assert vocab['program_token_to_idx']['<NULL>'] == 0
    assert vocab['program_token_to_idx']['<START>'] == 1
    assert vocab['program_token_to_idx']['<END>'] == 2
    return vocab


def load_scenes(scenes_json):
    with open(scenes_json) as f:
        scenes_dict = json.load(f)['scenes']
    scenes = []
    for s in scenes_dict:
        table = []
        for i, o in enumerate(s['objects']):
            # print(o)
            item = {}
            item['id'] = '%d-%d' % (s['image_index'], i)
            if '3d_coords' in o:
                item['3d_position'] = o['3d_coords']
                item['position'] = [np.dot(o['3d_coords'], s['directions']['right']),
                                    np.dot(o['3d_coords'], s['directions']['front']),
                                    o['3d_coords'][2]]
            else:
                item['position'] = o['position']
            
            if 'pixel_coords' in o:
                item['2d_coords'] = o['pixel_coords']

            item['color'] = o['color']
            item['material'] = o['material']
            item['shape'] = o['shape']
            item['size'] = o['size']
            table.append(item)
        scenes.append(table)
    return scenes
    

def load_embedding(path):
    return torch.Tensor(np.load(path))
