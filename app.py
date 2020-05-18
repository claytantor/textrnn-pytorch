from flask import Flask
from flask import jsonify

import sys, os
import os.path
from os import path
import torch 
import torch.nn as nn
import torch.nn.functional as F
import argparse
import glob
import random

import numpy as np
from collections import Counter
import os
import tensorflow as tf
from argparse import Namespace

from songdb import search_song_titles

import spacy

nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
app.config.from_envvar('APP_CONFIG')

def loadYaml(configFilePath):
    cfg = None
    with open(configFilePath, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg


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
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    return criterion, optimizer


def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5):
    net.eval()

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))
    
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_vocab[choice])

    for _ in range(100):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])

    return words


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

            state_h = state_h.detach()
            state_c = state_c.detach()

            loss_value = loss.item()

            # Perform back-propagation
            loss.backward(retain_graph=True)

            # Update the network's parameters
            optimizer.step()

            loss.backward()

            _ = torch.nn.utils.clip_grad_norm_(
                net.parameters(), flags.gradients_norm)

            optimizer.step()

            if iteration % 100 == 0:
                print('Epoch: {}/{}'.format(e, iteration_count),
                      'Iteration: {}'.format(iteration),
                      'Loss: {}'.format(loss_value))

            if iteration % 1000 == 0:
                ''.join(predict(device, net, flags.initial_words, n_vocab,
                        vocab_to_int, int_to_vocab, top_k=5))
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

def query_songs_and_write(songs_file_path):

    song_titles = search_song_titles()

    outF = open(songs_file_path, "w")
    for line in song_titles:
        # write line to output file
        outF.write(line+" ")
    
    outF.close()

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

def get_seed_file(seed_file_path):
    data = ''
    with open(seed_file_path, 'r') as file:
        data = file.read(40856).replace('\n', '')
    return data.split(" ")

@app.route('/title')
def get_song_title():

    session_id = os.environ.get("SESSION", default="song_titles_01")
    
    session_dir = os.path.join(os.getcwd(), "training/{}".format(session_id))
    checkpoint_path = "{}/checkpoint_pt".format(session_dir)

    word_seed = get_seed_file('training/{}/source/song_titles.txt'.format(session_id))
    position = random.randint(0, len(word_seed)-10)

    source_dir = "{}/source".format(session_dir)
    songs_title_source_file = "{}/song_titles.txt".format(source_dir)


    flags = Namespace(  
        train_file=songs_title_source_file,
        seq_size=32,
        batch_size=16,
        embedding_size=64,
        lstm_size=64,
        gradients_norm=5,
        initial_words=word_seed[position:position+4],
        predict_top_k=5,
        checkpoint_path=checkpoint_path,
    )


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print_info(device)
    
    int_to_vocab, vocab_to_int, n_vocab, in_text, out_text = get_data_from_file(
        flags.train_file, flags.batch_size, flags.seq_size)

    net = RNNModule(n_vocab, flags.seq_size,
                        flags.embedding_size, flags.lstm_size)


    list_of_files = glob.glob('{}/*'.format(checkpoint_path)) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)

    net.load_state_dict(torch.load(latest_file))
    net.eval()
    net = net.to(device)

    words = predict(device, net, flags.initial_words, n_vocab,
        vocab_to_int, int_to_vocab, top_k=5)

    doc = nlp(' '.join(words))

    # Analyze syntax
    nouns =  [chunk.text for chunk in doc.noun_chunks]
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
    adps = [token.lemma_ for token in doc if token.pos_ == "ADP"]
    propn = [token.lemma_ for token in doc if token.pos_ == "PROPN"]

    # NOUN NOUN VERB
    return jsonify({'title':capitalize_all_words('{} {} {}'.format(
        nouns[random.randint(0, len(nouns)-1)],           
        nouns[random.randint(0, len(nouns)-1)],
        verbs[random.randint(0, len(verbs)-1)]
        ))})


@app.route('/')
def hello():
    return 'Hello, World!'



if __name__ == '__main__':
    app.run()