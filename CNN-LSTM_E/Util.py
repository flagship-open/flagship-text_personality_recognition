import json
from tflearn.data_utils import pad_sequences


"""
A series of data manipulation functions
"""

#Loading vocabulary with embeddings from an embedding table file
"""
Input: id|||word|||embedding
Output
 1) ids2word[id]:word
 2) word2ids[word]:id
 3) ids2vec[id]:embedding
"""
def load_vocab(embedding_table_path):
    embedding_table = open(embedding_table_path).readlines()
    ids2word = dict()
    word2ids = dict()
    ids2vec = dict()
    for vocab_info in embedding_table:
        if vocab_info.strip() == "":
            continue

        vocab_info = vocab_info.split("|||")
        ids2word[int(vocab_info[0])] = vocab_info[1]
        word2ids[vocab_info[1]] = vocab_info[0]
        ids2vec[int(vocab_info[0])] = vocab_info[2]

    return ids2word, word2ids, ids2vec

#Loading train/dev/test dataset and converting them into input_format of network for sentence classification
"""
Output: sequence of word indices and sequence of class indices for each dataset (train, dev, or test dataset)
X = [
    //first dialogue: [word_1,1, ... word_1,max_utt_length],[word_2,1, ... word_2,max_utt_length], ... ,[word_max_dialog_length,1 ... word_max_dialog_length,max_utt_length],
                                    ...
    //last dialogue: [word_1,1, ... word_1,max_utt_length],[word_2,1, ... word_2,max_utt_length], ... ,[word_max_dialog_length,1 ... word_max_dialog_length,max_utt_length],
    ]
Y = [[class1,1, class1,2, ... class1,last_utt] 
                        ...
     [class_last_dialogue,1, ... class_last_dialgue,last_utt]]

len(X) == len(Y)

"""
def load_indices_sequence_with_label_sequence(json_data, word2ids, max_dialogue_length, max_utterance_length):
    input = []
    output = []
    mask = []
    sequence_length = []

    for dialogue in json_data:
        utts_list = []
        label_with_index = []
        mask_of_dialogue = []

        sequence_length.append(len(dialogue))

        for utterance in dialogue:
            utt_with_indices = []
            utt = utterance["input"].split()
            for word in utt:
                if word in word2ids:
                    utt_with_indices.append(word2ids[word])
                else:
                    utt_with_indices.append(word2ids["<UNK>"])
            utts_list.append(utt_with_indices)
            label_with_index.append(int(utterance["output"]))
            mask_of_dialogue.append(1)


        if len(utts_list) != max_dialogue_length:
            for i in range(max_dialogue_length-len(utts_list)):
                utts_list.append([0])

        for utt in utts_list:
            input.append(utt)

        output.append(label_with_index)
        mask.append(mask_of_dialogue)

    input = pad_sequences(input,maxlen=max_utterance_length,value=0)
    input = input.tolist()

    output = pad_sequences(output, maxlen=max_dialogue_length, value=0)
    output = output.tolist()

    mask = pad_sequences(mask, maxlen=max_dialogue_length, value=0)
    mask = mask.tolist()


    return input, output, sequence_length, mask

def load_data(data_info):
    #Loading word indices
    ids2word, word2ids, ids2vec = load_vocab(data_info['word_embedding_path'])
    ids2vec = []
    ids2word = []

    #Loading train/dev/test dataset
    train = json.load(open(data_info["train_data_path"],'r'))
    dev = json.load(open(data_info["dev_data_path"],'r'))
    test = json.load(open(data_info["test_data_path"],'r'))

    train_x, train_y, train_sequence_length, train_mask = load_indices_sequence_with_label_sequence(train, word2ids, data_info['max_dialogue_length'], data_info['max_utterance_length'])
    dev_x, dev_y, dev_sequence_length, dev_mask = load_indices_sequence_with_label_sequence(dev, word2ids, data_info['max_dialogue_length'], data_info['max_utterance_length'])
    test_x, test_y, test_sequence_length, test_mask = load_indices_sequence_with_label_sequence(test, word2ids, data_info['max_dialogue_length'], data_info['max_utterance_length'])

    return train_x, train_y, train_sequence_length, train_mask, dev_x, dev_y, dev_sequence_length, dev_mask, test_x, test_y, test_sequence_length, test_mask

#Generating batches in train dataset for one epoch

#To be confirmed
def load_batches(train_x, train_y, train_sequence_length, train_mask, hyperparam):
    if (len(train_x) / hyperparam['max_dialogue_length']) != len(train_y):
        print("Invalid!")

    train_x_batch = [train_x[i:i + (hyperparam['batch_size'] * hyperparam['max_dialogue_length'])] for i in range(0, len(train_x), (hyperparam['batch_size'] * hyperparam['max_dialogue_length']))]
    train_y_batch = [train_y[i:i + (hyperparam['batch_size'])] for i in range(0, len(train_y), hyperparam['batch_size'])]
    train_mask_batch = [train_mask[i:i + (hyperparam['batch_size'])] for i in range(0, len(train_mask), hyperparam['batch_size'])]
    train_seq_length_batch = [train_sequence_length[i:i + (hyperparam['batch_size'])] for i in range(0, len(train_sequence_length), hyperparam['batch_size'])]

    return train_x_batch, train_y_batch, train_seq_length_batch, train_mask_batch

def load_batches_test(train_x, train_y, train_sequence_length, train_mask, hyperparam):
    if (len(train_x) / hyperparam['max_dialogue_length']) != len(train_y):
        print("Invalid!")

    train_x_batch = [train_x[i:i + (500 * hyperparam['max_dialogue_length'])] for i in range(0, len(train_x), (500 * hyperparam['max_dialogue_length']))]
    train_y_batch = [train_y[i:i + 500] for i in range(0, len(train_y), 500)]
    train_mask_batch = [train_mask[i:i + 500] for i in range(0, len(train_mask), 500)]
    train_seq_length_batch = [train_sequence_length[i:i + 500] for i in range(0, len(train_sequence_length), 500)]

    return train_x_batch, train_y_batch, train_seq_length_batch, train_mask_batch



