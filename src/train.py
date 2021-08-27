import sys, os
import os.path
from os import path
import torch 
import torch.nn as nn
import torch.nn.functional as F
import argparse
import glob
import random
import shutil

import numpy as np
from collections import Counter
import os
from argparse import Namespace

# from songdb import search_song_titles

import spacy

nlp = spacy.load("en_core_web_sm")


class RNNModule(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(RNNModule, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))



def get_data_from_file(train_file, batch_size, seq_size):

    with open(train_file, 'r') as f:
        text = f.read()

    text = text.split()

    word_counts = Counter(text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {k: w for k, w in enumerate(sorted_vocab)}
    vocab_to_int = {w: k for k, w in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)

    print('Vocabulary size', n_vocab)

    int_text = [vocab_to_int[w] for w in text]
    num_batches = int(len(int_text) / (seq_size * batch_size))
    in_text = int_text[:num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = in_text[0]
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    return int_to_vocab, vocab_to_int, n_vocab, in_text, out_text

def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for i in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, i:i+seq_size], out_text[:, i:i+seq_size]


def get_loss_and_train_op(net, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    
    #criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

    return criterion, optimizer



def train(device, net, criterion, optimizer,  in_text, out_text, n_vocab, vocab_to_int,  int_to_vocab, flags, target_dir, iteration_count=200):

    iteration = 0

    for e in range(iteration_count):
        batches = get_batches(in_text, out_text, flags.batch_size, flags.seq_size)
        state_h, state_c = net.zero_state(flags.batch_size)
        
        # Transfer data to GPU
        state_h = state_h.to(device)
        state_c = state_c.to(device)
        for x, y in batches:
            iteration += 1
            
            # Tell it we are in training mode
            net.train()

            # Reset all gradients
            optimizer.zero_grad()

            # Transfer data to GPU
            x = torch.tensor(x).to(device)
            y = torch.tensor(y).to(device)

            logits, (state_h, state_c) = net(x, (state_h, state_c))
            loss = criterion(logits.transpose(1, 2), y)
     
            loss_value = loss.item()

            loss.backward()

            state_h = state_h.detach()
            state_c = state_c.detach()

            _ = torch.nn.utils.clip_grad_norm_(
                net.parameters(), flags.gradients_norm)

            optimizer.step()

            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(e, iteration_count),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))
            
            if iteration % 1000 == 0:
                torch.save(net.state_dict(),
                           '{}/model-{}.pth'.format(target_dir, iteration))


def print_info(device):
    print('__Python VERSION:', sys.version)
    print('__pyTorch VERSION:', torch.__version__)
    print('__CUDA AVILABLE:', torch.cuda.is_available())
    print('__CUDNN VERSION:', torch.backends.cudnn.version())
    print('__Number CUDA Devices:', torch.cuda.device_count())
    print('__Device:', device, torch.cuda.get_device_name(device))
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print ('Available devices ', torch.cuda.device_count())
    print ('Current cuda device ', torch.cuda.current_device())


def make_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError:
        print("Cannot create dir error:", sys.exc_info()[0])
        raise

def capitalize_all_words(all_words):
    capitalized = []
    words = all_words.split(' ')
    for word in words:
        capitalized.append(word.capitalize())
    return ' '.join(capitalized)

def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--initial", action="store", default="I am",
        required=False, dest="initial", help="Initial words to seed") 

    parser.add_argument("-s", "--session", action="store",
        required=True,  dest="session", help="the sessionid for the training")    

    parser.add_argument("-n", "--number", action="store", default=200,
        required=False,  dest="number", help="the number of iterations")                                        
        
    parser.add_argument("-f", "--file", action="store",
        required=False, dest="file", help="Source file") 

    parser.add_argument("-w", "--words", action="store", default="5",
        required=False, dest="words", help="Number of words")   
    
    parser.add_argument("-o", "--out", action="store",
        required=False, dest="outfile", help="save predictions to file name")   

    parser.add_argument("-d", "--device", action="store",
        required=False, dest="device", default="cuda", help="use the targeted cuda device")   

    parser.add_argument("-l", "--lr", action="store",
        required=False, dest="learning_rate", default="0.001", help="learning rate to use")    

    args = parser.parse_args()

    print(args.device)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print_info(device)
    
    session_dir = os.path.join(os.getcwd(), "/workspace/training/{}".format(args.session))
    make_dir(session_dir)

    checkpoint_path = "{}/checkpoint_pt".format(session_dir)
    make_dir(checkpoint_path)

    source_dir = "{}/source".format(session_dir)
    make_dir(source_dir)
    # copy to song titles 
    if path.exists(args.file):
        songs_title_source_file = "{}/source_lines.txt".format(source_dir)
        shutil.copyfile(args.file, songs_title_source_file)
    else:
        raise ValueError('cannot find input file: {}'.format(args.file))


    flags = Namespace(  
            train_file=songs_title_source_file,
            seq_size=32,
            batch_size=16,
            embedding_size=64,
            lstm_size=64,
            gradients_norm=5,
            initial_words=['I', 'am'],
            predict_top_k=5,
            checkpoint_path=checkpoint_path,
        )

    
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(
        flags.train_file, flags.batch_size, flags.seq_size)

    net = RNNModule(n_vocab, flags.seq_size,
                    flags.embedding_size, flags.lstm_size)

                    
    net = net.to(device)

    criterion, optimizer = get_loss_and_train_op(net, float(args.learning_rate))

    train(device, net, criterion, optimizer,  in_text, out_text, n_vocab, vocab_to_int, int_to_vocab, flags, checkpoint_path, iteration_count=int(args.number))





if __name__ == "__main__":
    main(sys.argv[1:]) 
