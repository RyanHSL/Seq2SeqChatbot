import json
import os
import re
import io
import unicodedata

from config import getConfig
from string import punctuation
from keras.preprocessing.text import Tokenizer


configs = {}
configs = getConfig.get_config()
max_vocab_size = configs['max_num_words']
conv_path = configs['resource_data']
vocab_inp_path = configs['vocab_inp_path']
vocab_tar_path = configs['vocab_tar_path']
vocab_inp_size = configs['vocab_inp_size']
vocab_tar_size = configs['vocab_tar_size']
seq_train = configs['seq_data']

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')

def preprocess_util():
    if not os.path.exists(conv_path):
        print("Cannot find the training data. Please make sure it is in the train_data folder.")
        exit()
    file = open(os.getcwd() + '/Data/train_data/seq.data','r').read()
    qna_list = [f.split('\t') for f in file.split('\n')]

    questions = [x[0] for x in qna_list]
    answers = [x[1] for x in qna_list]

    return questions, answers

def create_vocab(lang, vocab_path, vocab_size=max_vocab_size):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=3)
    tokenizer.fit_on_texts(lang)
    vocab = json.loads(tokenizer.to_json(ensure_ascii=False))
    vocab['index_word'] = tokenizer.index_word
    vocab['word_index'] = tokenizer.word_index
    vocab['document_count']=tokenizer.document_count
    vocab = json.dumps(vocab, ensure_ascii=False)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write(vocab)
    f.close()
    print("Dict is saved to:{}".format(vocab_path))

def preprocess_sentence(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)

    s = re.sub(r"[^a-zA-Z?.!,¿]+", " ", s)
    s = s.strip()

    s ='<start> '+ s + ' <end>'

    return s

lines = io.open(seq_train, encoding='UTF-8').readlines()
word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines]
input_lang, target_lang = zip(*word_pairs)
questions, answers = preprocess_util()
pre_questions = [preprocess_sentence(w) for w in questions]
pre_answers = [preprocess_sentence(w) for w in answers]
create_vocab(input_lang,vocab_inp_path,vocab_inp_size)
create_vocab(target_lang,vocab_tar_path,vocab_tar_size)
