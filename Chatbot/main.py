import json
import os
import sys
import time
import tensorflow as tf
# import horovod.tensorflow as hvd
import seq2seqModel
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from config import getConfig
from data_util import preprocess_util, preprocess_sentence
import io

# hvd.init()
configs = getConfig.get_config()
vocab_inp_size = configs['vocab_inp_size']
vocab_tar_size = configs['vocab_tar_size']
embedding_dim = configs['embedding_dim']
units = configs['layer_size']
BATCH_SIZE = configs['batch_size']
max_length_inp=configs['max_sequence_length']
max_length_tar=configs['max_sequence_length']

log_dir=configs['log_dir']
writer = tf.summary.create_file_writer(log_dir)
questions, answers = preprocess_util()
pre_questions = [preprocess_sentence(w) for w in questions]
pre_answers = [preprocess_sentence(w) for w in answers]

def read_data(path):
    path = os.getcwd() + '/' + path
    if not os.path.exists(path):
        path=os.path.dirname(os.getcwd())+'/'+ path
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines]
    input_lang,target_lang=zip(*word_pairs)
    input_tokenizer=tokenize(configs['vocab_inp_path'])
    target_tokenizer=tokenize(configs['vocab_tar_path'])
    input_tensor=input_tokenizer.texts_to_sequences(input_lang)
    target_tensor=target_tokenizer.texts_to_sequences(target_lang)
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor, maxlen=max_length_inp,
                                                           padding='post')
    target_tensor= tf.keras.preprocessing.sequence.pad_sequences(target_tensor, maxlen=max_length_tar,
                                                           padding='post')
    return input_tensor,input_tokenizer,target_tensor,target_tokenizer

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='OOV')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer

def load_dataset(data, num_examples=None):
    # creating cleaned input, output pairs
    if(num_examples != None):
        targ_lang, inp_lang, = data[:num_examples]
    else:
        targ_lang, inp_lang, = data

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

num_examples = configs['num_examples']
data = pre_answers, pre_questions
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(data, num_examples)
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
BUFFER_SIZE = len(input_tensor_train)
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

def convert(lang, tensor):
    for t in tensor:
        if t!=0:
            print ("%d ----> %s" % (t, lang.index_word[t]))

convert(inp_lang, input_tensor_train[0])
convert(targ_lang, target_tensor_train[0])
# input_tensor, input_token, target_tensor, target_token = read_data(configs['seq_data'])
steps_per_epoch = len(input_tensor) // (configs['batch_size'])
BUFFER_SIZE = len(input_tensor)
dataset = tf.data.Dataset.from_tensor_slices((input_tensor,target_tensor)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
enc_hidden = seq2seqModel.encoder.initialize_hidden_state()
# dataset = dataset.shard(hvd.size(), hvd.rank())

def train():
    print("Preparing data in %s" % configs['train_data'])
    print('Step per epoch: {}'.format(steps_per_epoch))
    checkpoint_dir = configs['model_data']
    ckpt = None
    try:
        ckpt=tf.io.gfile.listdir(checkpoint_dir)
    except:
        print('Cannot find pretrained models. Start training the new model.')
    if ckpt:
        print("reload pretrained model")
        seq2seqModel.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    start_time = time.time()
    #current_loss=2
    #min_loss=configs['min_loss']
    epoch = 0
    train_epoch = configs['epochs']
    losses = []

    while epoch<train_epoch:
        start_time_epoch = time.time()
        enc_hidden = seq2seqModel.encoder.initialize_hidden_state()
        total_loss = 0
        for (batch,(inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = seq2seqModel.training_step(inp, targ, targ_lang, enc_hidden)
            total_loss += batch_loss
            print('epoch:{}batch:{} batch_loss: {}'.format(epoch,batch,batch_loss))

        step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
        step_loss = total_loss / steps_per_epoch
        current_steps = +steps_per_epoch
        epoch_time_total = (time.time() - start_time)
        print('current step: {} total time: {}  time per epoch: {} loss per step {:.4f}'
              .format(current_steps, epoch_time_total, step_time_epoch, step_loss))
        losses.append(step_loss)
        seq2seqModel.checkpoint.save(file_prefix=checkpoint_prefix)
        sys.stdout.flush()
        epoch = epoch + 1
        with writer.as_default():
            tf.summary.scalar('loss', step_loss, step=epoch)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def remove_tags(sentence):
    return sentence.split("<start>")[-1].split("<end>")[0]

def predict(sentence):
    input_tokenizer = tokenize(configs['vocab_inp_path'])
    target_tokenizer = tokenize(configs['vocab_tar_path'])
    checkpoint_dir = configs['model_data']
    seq2seqModel.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    sentence = preprocess_sentence(sentence)
    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)
    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = seq2seqModel.encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = seq2seqModel.decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        attention_weights = tf.reshape(attention_weights, (-1,))

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return remove_tags(result), remove_tags(sentence)

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return remove_tags(result), remove_tags(sentence)
'''
Change the mode in the parameter.ini to 'train' to train the model.
Change the mode in the parameter.ini to 'serve' to talk to the chatbot.
'''
if __name__ == '__main__':
    if len(sys.argv) - 1:
        configs = getConfig.get_config(sys.argv[1])
    else:
        configs = getConfig.get_config()
    print('\n>> Mode : %s\n' %(configs['mode']))
    if configs['mode'] == 'train':
        print('training mode')
        train()
    elif configs['mode'] == 'serve':
        print('serving mode')
        print('Hi there. This is Rui''s chatbot. Please say something.')
        while True:
            sentence = str(input('Enter your sentence or enter exit to exit.\n'))
            if sentence == 'exit':
                break
            try:
                answer, result = predict(sentence)
                print(answer)
            except:
                print('The word is out of vocab. Please try again.')