import Model, Util, Hparam
import tensorflow as tf
import json
import collections
from tflearn.data_utils import pad_sequences
from konlpy.tag import Komoran
import argparse
import os
import numpy as np

# python3 Eval.py test.json

def load_data(input_path, hyperparameter,analyzer):
    json_data = json.load(open(input_path, 'r'))

    ids2word, word2ids, ids2vec = Util.load_vocab(hyperparameter['word_embedding_path'])

    input = []
    mask = []
    sequence_length = []

    for dialogue in json_data:
        utts_list = []
        mask_of_dialogue = []
        sequence_length.append(len(dialogue))

        for utterance in dialogue:
            utt_with_indices = []

            utt_with_pos = analyzer.pos(utterance["utterance"])
            new_utt = ""
            for token in utt_with_pos:
                new_token = token[0] + "/" + token[1] + " "
                new_utt += new_token
            new_utt = new_utt.strip()
            utt = new_utt.split()

            for word in utt:
                if word in word2ids:
                    utt_with_indices.append(word2ids[word])
                else:
                    utt_with_indices.append(word2ids["<UNK>"])
            utts_list.append(utt_with_indices)
            mask_of_dialogue.append(1)


        if len(utts_list) != hyperparameter['max_dialogue_length']:
            for i in range(hyperparameter['max_dialogue_length']-len(utts_list)):
                utts_list.append([0])

        for utt in utts_list:
            input.append(utt)

        mask.append(mask_of_dialogue)

    input = pad_sequences(input,maxlen=hyperparameter['max_utterance_length'],value=0)
    input = input.tolist()

    mask = pad_sequences(mask, maxlen=hyperparameter['max_dialogue_length'], value=0)
    mask = mask.tolist()

    return input, sequence_length, mask

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str,
                help="input path")

    args = parser.parse_args()
    input_path = args.input_path

    hyperparam = Hparam.get_hparam()
    model = Model.Personality_Recognizer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        #Loading inputs
        #Morphological anayler
        komoran = Komoran()

        #Users can change the path of input
        input_data, sequence_length, mask  = load_data(input_path=input_path, hyperparameter=hyperparam, analyzer=komoran)
        print("#Loading the model ...")
        import sys
        all_results = []
        for data_type in ['A', 'C', 'E', 'N', 'O']:
            ckpt = tf.train.get_checkpoint_state(os.path.join('CNN-LSTM_{}'.format(data_type), 'Parameter'))
            saver.restore(sess, ckpt.model_checkpoint_path)
            predict = sess.run(tf.argmax(model.softmax_logits, -1), feed_dict={model.input_x: input_data,
                                                            model.lstm_seq_length: sequence_length,
                                                            model.mask: mask,
                                                            model.dropout_keep_prob: hyperparam['test_dropout_keep_prob']})

            results = []
            with open(os.path.join('Eval_Gold', 'Data_{}'.format(data_type),'test_{}.json'.format(data_type))) as f:
                gold = json.load(f)
                for instances, predict_instances in zip(gold, predict):
                    predict_instances = predict_instances[:len(instances)]
                    result = []
                    for instance, predict_instance in zip(instances, predict_instances):
                        result_instance = {}
                        result_instance['predict'] = predict_instance
                        result_instance['gold'] = instance['output']
                        result_instance['utterance'] = instance['utterance']
                        result.append(result_instance)
                    results.append(result)
            all_results.append(results)


    type_accs = []
    examples = []
    with open('result.tsv', 'w') as f:
        for A_ins, C_ins, E_ins, N_ins, O_ins in zip(*all_results):
            f.write('doc\tO_gold\tO_pred\tC_gold\tC_pred\tE_gold\tE_pred\tA_gold\tA_pred\tN_gold\tN_pred\n')
            instance_acc = []
            type_acc = []
            flag_1 = False
            flag_2 = False
            for A_in, C_in, E_in, N_in, O_in in zip(A_ins, C_ins, E_ins, N_ins, O_ins):
                f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(A_in['utterance'], 
                    O_in['gold'], O_in['predict'],
                    C_in['gold'], C_in['predict'], E_in['gold'], E_in['predict'], 
                    A_in['gold'], A_in['predict'], N_in['gold'], N_in['predict']))
                type_acc.append([int(O_in['gold'] == O_in['predict']), 
                    int(C_in['gold'] == C_in['predict']), int(E_in['gold'] == E_in['predict']), 
                    int(A_in['gold'] == A_in['predict']), int(N_in['gold'] == N_in['predict'])])
                instance_acc.append([int(O_in['gold'] == O_in['predict']), 
                    int(C_in['gold'] == C_in['predict']), int(E_in['gold'] == E_in['predict']), 
                    int(A_in['gold'] == A_in['predict']), int(N_in['gold'] == N_in['predict'])])

                if (A_in['gold'] == 1) and (A_in['predict'] == 1):
                    flag_1 = True
                if (C_in['gold'] == 1) and (C_in['predict'] == 1):
                    flag_2 = True
                if (E_in['gold'] == 1) and (E_in['predict'] == 1):
                    flag_2 = True
                if (O_in['gold'] == 1) and (O_in['predict'] == 1):
                    flag_2 = True
                if (N_in['gold'] == 1) and (N_in['predict'] == 1):
                    flag_2 = True
            type_accs.append(np.mean(type_acc, 0))
            if np.min(np.mean(instance_acc, 0)) > 0.85 and flag_1 and flag_2 and len(examples) < 3:
                example = []
                for idx, A_in in enumerate(A_ins): 
                    instance = {}
                    instance['utterance'] = A_in['utterance']
                    instance['id'] = str(idx)
                    example.append(instance)
                examples.append(example)

    with open('example.json', 'w') as f:
        json.dump(examples, f, indent='\t', ensure_ascii=False) 
    type_accs = np.mean(type_accs, 0).tolist()
    print('O: {:.3f}, C: {:.3f}, E: {:.3f}, A: {:.3f}, N: {:.3f}, Avg: {:.3f}'.format(type_accs[0], 
        type_accs[1], type_accs[2], type_accs[3], type_accs[4], np.mean(type_accs)))

