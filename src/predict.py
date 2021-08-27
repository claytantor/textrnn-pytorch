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
import uuid

import numpy as np
from collections import Counter
import os
from argparse import Namespace

# from songdb import search_song_titles

import spacy

nlp = spacy.load("en_core_web_sm")

import nltk
import codecs
nltk.download('home')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('stopwords')

load_grammar = nltk.data.load('file:src/english_grammar.cfg')


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

    #print('Vocabulary size', n_vocab)

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


def predict(device, net, words, n_vocab, vocab_to_int, int_to_vocab, top_k=5, predict_length=100):
    net.eval()

    state_h, state_c = net.zero_state(1)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    #print(words)
    for w in words:
        ix = torch.tensor([[vocab_to_int[w]]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))
    
    _, top_ix = torch.topk(output[0], k=top_k)
    choices = top_ix.tolist()
    choice = np.random.choice(choices[0])

    words.append(int_to_vocab[choice])

    for _ in range(predict_length):
        ix = torch.tensor([[choice]]).to(device)
        output, (state_h, state_c) = net(ix, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=top_k)
        choices = top_ix.tolist()
        choice = np.random.choice(choices[0])
        words.append(int_to_vocab[choice])

    return words



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

# def build_grammar(subject_tokens, verb_tokens, part_tokens, all_tokens):

#     # print(subject_tokens, verb_tokens, part_tokens)

#     if(len(subject_tokens)>0 and len(verb_tokens)>0 and len(part_tokens)>0):
#         w_subject = subject_tokens[random.randrange(len(subject_tokens))]
#         w_verb = verb_tokens[random.randrange(len(verb_tokens))]
#         w_part = part_tokens[random.randrange(len(part_tokens))]
    
#         return "{} {} {}".format(w_subject['word'], w_verb['word'], w_part['word'])

def add_if_avail(sentence, sentences):
    if(sentence != None):
        sentences.append(sentence)



def sentence_ok(tokens):

    print(tokens)

    # remove one letter nouns
    tokens = list(filter(lambda x: x['word']!='and' and x['word']!='or', tokens))

    # 'good' | 'bad' | 'beautiful' | 'innocent'
    adj_list = list(filter(lambda x: x['pos'] == 'ADJ' and x['tag'] == 'JJ', tokens))
    adj_grammar = " | ".join(set(list(map(lambda x: "'{}'".format(x['word'].lower()), adj_list))))
    print("ADJ", adj_grammar)

    adv_list = list(filter(lambda x: x['pos'] == 'ADV', tokens))
    adv_grammar = " | ".join(set(list(map(lambda x: "'{}'".format(x['word'].lower()), adv_list))))
    print("ADV", adv_grammar)

    # 'saw' | 'liked' | 'ate' | 'shot'
    tv_past_list = list(filter(lambda x: x['pos'] == 'VERB' and x['tag'] =='VBD', tokens))
    tv_past_grammar = " | ".join(set(list(map(lambda x: "'{}'".format(x['word'].lower()), tv_past_list))))
    print("TV_Past", tv_past_grammar)

    # 'dissappear' | 'walk'
    iv_pres_pl_list = list(filter(lambda x: x['pos'] == 'VERB' and x['tag'] =='VBG', tokens))
    iv_pres_pl_grammar = " | ".join(set(list(map(lambda x: "'{}'".format(x['word'].lower()), iv_pres_pl_list))))
    print("iv_pres_pl_grammar", iv_pres_pl_grammar)

    # 'dog' | 'girl' | 'car' | 'child' | 'apple' | 'elephant'
    N_Sg_list = list(filter(lambda x: x['pos'] == 'NOUN' and x['tag'] =='NN', tokens))
    N_Sg_grammar = " | ".join(set(list(map(lambda x: "'{}'".format(x['word'].lower()), N_Sg_list))))
    print("N_Sg_grammar", N_Sg_grammar)

    #{'word': 'nephalot', 'pos': 'PROPN', 'dep': 'nmod', 'tag': 'NNP'}
    PropN_Sg_list = list(filter(lambda x: x['pos'] == 'PROPN' and x['tag'] =='NNP', tokens))
    PropN_Sg_grammar = " | ".join(set(list(map(lambda x: "'{}'".format(x['word'].lower()), PropN_Sg_list))))
    print("PropN_Sg_grammar", PropN_Sg_grammar)

    #{'word': 'beaches', 'pos': 'VERB', 'dep': 'compound', 'tag': 'VBZ'}
    IV_Pres_Sg_list = list(filter(lambda x: x['pos'] == 'VERB' and x['tag'] =='VBZ', tokens))
    IV_Pres_Sg_grammar = " | ".join(set(list(map(lambda x: "'{}'".format(x['word'].lower()), IV_Pres_Sg_list))))
    print("IV_Pres_Sg_grammar", IV_Pres_Sg_grammar)


    #PropN_Pl {'word': 'I', 'pos': 'PRON', 'dep': 'nsubj', 'tag': 'PRP'}

    # Det_Sg {'word': 'every', 'pos': 'DET', 'dep': 'dobj', 'tag': 'DT'}

    if(len(adj_list)>0 and len(adv_list)>0):

        g = nltk.CFG.fromstring("""
            S -> NP_Sg VP_Sg | NP_Pl VP_Pl
            NP -> NP_Pl      | NP_Sg
            NP_Sg ->       N_Sg | Det_Sg N_Sg | Det_Both N_Sg | Adj N_Sg | Det_Sg Adj N_Sg | Det_Both Adj N_Sg| PropN_Sg
            NP_Pl ->       N_Pl | Det_Pl N_Pl | Det_Both N_Pl | Adj N_Pl | Det_Pl Adj N_Pl | Det_Both Adj N_Pl| PropN_Pl
            VP_Sg -> IV_Pres_Sg | IV_Past     | TV_Pres_Sg    | TV_Past  | TV_Pres_Sg NP   | TV_Past NP       | Adv IV_Pres_Sg | Adv IV_Past | Adv TV_Pres_Sg NP | Adv TV_Past NP
            VP_Pl -> IV_Pres_Pl | IV_Past     | TV_Pres_Pl    | TV_Past  | TV_Pres_Pl NP   | TV_Past NP       | Adv IV_Pres_Pl | Adv IV_Past | Adv TV_Pres_Pl NP | Adv TV_Past NP
            N_Pl -> 'girls' | 'boys' | 'children' | 'cars' | 'apples' | 'dogs'
            Adj -> {adj_grammar}
            Adv -> {adv_grammar}
            N_Sg -> {N_Sg_grammar}
            PropN_Sg -> {PropN_Sg_grammar}
            PropN_Pl -> 'they'  | 'i'
            Det_Sg -> 'this' | 'every' | 'a' | 'an'
            Det_Pl -> 'these' | 'all'
            Det_Both -> 'some' | 'the' | ' several'
            IV_Pres_Sg -> {IV_Pres_Sg_grammar}
            TV_Pres_Sg -> 'sees' | 'likes' |'eat'
            IV_Pres_Pl -> {iv_pres_pl_grammar}
            TV_Pres_Pl ->'see' | 'like'
            IV_Past -> 'dissappeared' | 'walked'
            TV_Past -> {tv_past_grammar}
        """.format(
            adj_grammar=adj_grammar, 
            tv_past_grammar=tv_past_grammar,
            adv_grammar=adv_grammar,
            N_Sg_grammar=N_Sg_grammar,
            PropN_Sg_grammar=PropN_Sg_grammar,
            IV_Pres_Sg_grammar=IV_Pres_Sg_grammar,
            iv_pres_pl_grammar=iv_pres_pl_grammar))


        sent_split = list(map(lambda x: x['word'], tokens))
        print(sent_split) 

        wrong_syntax=1
        rd_parser = nltk.RecursiveDescentParser(g)
        for tree_struc in rd_parser.parse(sent_split):
            s = tree_struc
            wrong_syntax=0
            print("Correct Grammer !!!")
            print(str(s))
            # f = open("demoEnglish.txt", "a")
            # f.write("Correct Grammer!!!!!")
            # f.write(str(s))
            # f.close()
        if wrong_syntax==1:
            print("Wrong Grammer!!!!!!")
            # f = open("demoEnglish.txt", "a")
            # f.write("Wrong Grammer!!!!!")
            # f.close()
    else:
        print("Wrong Grammer 2 !!!!!!")  

def rebuild_sentence(tokens):
    # print(tokens)
    

    """ Most sentences in English are constructed using one of the following five patterns:
         Subject–Verb–Object.
         Subject–Verb–Adjective.
         Subject–Verb–Adverb.
         Subject–Verb–Noun.
    """
    # sentence_ok(tokens)

    subject_tokens = list(filter(lambda x: x['dep'] == 'nsubj', tokens))

    # capitalize subjects
    for token in tokens:
        if token['pos'] == 'PROPN' and token['tag'] == 'NNP':
            token['word'] =  token['word'].capitalize() 

    verb_tokens = list(filter(lambda x: x['pos'] == 'VERB', tokens))

    # adj_tokens = list(filter(lambda x: x['pos'] == 'ADJ', tokens)) 
    # adv_tokens = list(filter(lambda x: x['pos'] == 'ADV', tokens)) 
    # noun_tokens = list(filter(lambda x: x['pos'] == 'NOUN', tokens)) 
    # obj_tokens = list(filter(lambda x: x['dep'] == 'nobj', tokens)) 


    if(len(subject_tokens)>0 and len(verb_tokens)>0):
        sent = ' '.join(list(map(lambda x: x['word'], tokens[:9])))        
        return sent if len(sent)<=42 else None
    else:
        return None

    # sentences = []
    # add_if_avail(build_grammar(subject_tokens, verb_tokens, adj_tokens, tokens), sentences)
    # add_if_avail(build_grammar(subject_tokens, verb_tokens, adv_tokens, tokens), sentences)
    # add_if_avail(build_grammar(subject_tokens, verb_tokens, noun_tokens, tokens), sentences)
    # add_if_avail(build_grammar(subject_tokens, verb_tokens, obj_tokens, tokens), sentences)

    # if(len(sentences)>0):
    #     # return sentences[random.randrange(len(sentences))]
    #     return sentences
    # else:
    #     return None


def adjust_sentences_nlp(sentences):

    result_nlp = []

    for sentence in sentences:
        all_stopwords = nlp.Defaults.stop_words
        text_tokens = nlp(sentence)
        tokens_without_sw= [word for word in text_tokens if not word in all_stopwords]

        pos_list = []
        for token in tokens_without_sw:        
            pos_list.append({"word":token.text, "pos":token.pos_, "dep": token.dep_, "tag": token.tag_ })

        # filter out punc
        pos_list = list(filter(lambda x: x['pos'] != "PUNCT", pos_list))
        # print(pos_list)

        sentence_rebuilt = rebuild_sentence(pos_list)
        if(sentence_rebuilt != None):
           result_nlp.append(sentence_rebuilt) 

    return result_nlp

def predict_words_too(training_path, session_name, predict_length, initial_words, device, top_k=5):
    session_dir = "{}/{}".format(training_path, session_name)
    checkpoint_path = "{}/checkpoint_pt".format(session_dir)
    source_dir = "{}/source".format(session_dir)
    songs_title_source_file = "{}/source_lines.txt".format(source_dir)
    
    flags = Namespace(  
            train_file=songs_title_source_file,
            seq_size=32,
            predict_length=int(predict_length),
            batch_size=16,
            embedding_size=64,
            lstm_size=64,
            gradients_norm=5,
            initial_words=initial_words.split(' '),
            predict_top_k=top_k,
            checkpoint_path=checkpoint_path,
        )
    
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
        vocab_to_int, int_to_vocab, top_k=top_k, predict_length=flags.predict_length )

    return words

def predict_words(args, device):
    session_dir = "{}/{}".format(args.training_path, args.session)
    checkpoint_path = "{}/checkpoint_pt".format(session_dir)
    source_dir = "{}/source".format(session_dir)
    songs_title_source_file = "{}/source_lines.txt".format(source_dir)
    
    flags = Namespace(  
            train_file=songs_title_source_file,
            seq_size=32,
            predict_length=int(args.predict),
            batch_size=16,
            embedding_size=64,
            lstm_size=64,
            gradients_norm=5,
            initial_words=args.initial.split(' '),
            predict_top_k=int(args.words),
            checkpoint_path=checkpoint_path,
        )
    
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
        vocab_to_int, int_to_vocab, top_k=5, predict_length=flags.predict_length )

    return words


def parse_sentences(words):

    doc = nlp(' '.join(words))
    sentences = [sent.text for sent in doc.sents]    
    return sentences


def make_sentences(args, device):

    words = predict_words(args, device)    
    sentences = parse_sentences(words)

    max_len = int(args.lines)
    simple_sentences = list(filter(lambda x: len(x.split(" ")) > 3 
        and len(x.split(" ")) < 10 
        and len(x)<=36, sentences))

    nlp_sentences = adjust_sentences_nlp(simple_sentences)[:max_len]

    return nlp_sentences

def make_sentences_too(args, device):

    words = predict_words(args, device)    
    sentences = parse_sentences(words)

    max_len = int(args.lines)
    simple_sentences = list(filter(lambda x: len(x.split(" ")) > 4, sentences))
    nlp_sentences = adjust_sentences_nlp(simple_sentences)[:max_len]

    return nlp_sentences

def predict_text(device_type, training_path, session_name, predict_length, initial_words, max_len=10):
    device = torch.device(device_type if torch.cuda.is_available() else 'cpu')
    words = predict_words_too(training_path, session_name, predict_length, initial_words, device, top_k=5)

    sentences = parse_sentences(words)

    # max_len = int(args.lines)
    simple_sentences = list(filter(lambda x: len(x.split(" ")) > 3 
        and len(x.split(" ")) < 10 
        and len(x)<=36, sentences))

    nlp_sentences = adjust_sentences_nlp(simple_sentences)[:max_len]
    print(nlp_sentences)

    return nlp_sentences

    
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
    
    parser.add_argument("-p", "--predict", action="store", default="100",
        required=False, dest="predict", help="Number of words to predict") 

    parser.add_argument("-l", "--lines", action="store", default="6",
        required=False, dest="lines", help="max lines") 

    parser.add_argument("-o", "--out_dir", action="store",
        required=False, dest="out_dir", help="save predictions to directory")   

    parser.add_argument("-d", "--device", action="store",
        required=False, dest="device", default="cuda", help="use the targeted cuda device")   

    parser.add_argument("-t", "--training_path", action="store", default="./workspace/training",
        required=False, dest="training_path", help="Number of words to predict") 


    parser.add_argument("-c", "--count", action="store",
        required=False, dest="count", default="1", help="how many to generate")          

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print_info(device)
        
    print(" ---- ")

    for index_val in range(0, int(args.count)):
        nlp_sentences = make_sentences_too(args, device)[1:]

        # dont write small poems
        if(args.out_dir != None and len(nlp_sentences)>=int(args.lines)-2):
        # if(args.out_dir != None):    
            try:
                os.makedirs(args.out_dir)
            except OSError:
                pass
            o_fn = "{}/{}_{}.txt".format(
                args.out_dir, 
                args.initial.replace(" ","_"), 
                str(uuid.uuid4()).replace('-','')[:8])   
            print(o_fn)
            out_file = open(o_fn,"w+")
            out_file.write('\n'.join(nlp_sentences))
            out_file.close()
        else:
            print("------------")
            for sentence in nlp_sentences:
                print("*",sentence)


if __name__ == "__main__":
    main(sys.argv[1:]) 
