import Model, Util, Hparam
import tensorflow as tf
import json
import collections
from tflearn.data_utils import pad_sequences
from konlpy.tag import Komoran
import argparse

def load_data(input_path, hyperparameter,analyzer):
    json_data = json.loads(input_path)

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

def write_result(input_path, output_path, prediction):
    json_data = json.loads(input_path)

    for i in range(len(json_data)):
        for j in range(len(json_data[i])):
            utt = json_data[i][j]["utterance"]
            spk = json_data[i][j]["speaker"]

            json_data[i][j]=collections.OrderedDict()
            json_data[i][j]["utterance"] = utt
            json_data[i][j]["speaker"] = spk

            json_data[i][j]["distribution_o"] = collections.OrderedDict()
            json_data[i][j]["distribution_o"]["50001"] = str(prediction[0][i][j][0])
            json_data[i][j]["distribution_o"]["50002"] = str(prediction[0][i][j][1])
            json_data[i][j]["distribution_o"]["50003"] = str(prediction[0][i][j][2])

            json_data[i][j]["distribution_c"] = collections.OrderedDict()
            json_data[i][j]["distribution_c"]["50004"] = str(prediction[1][i][j][0])
            json_data[i][j]["distribution_c"]["50005"] = str(prediction[1][i][j][1])
            json_data[i][j]["distribution_c"]["50006"] = str(prediction[1][i][j][2])

            json_data[i][j]["distribution_e"] = collections.OrderedDict()
            json_data[i][j]["distribution_e"]["50007"] = str(prediction[2][i][j][0])
            json_data[i][j]["distribution_e"]["50008"] = str(prediction[2][i][j][1])
            json_data[i][j]["distribution_e"]["50009"] = str(prediction[2][i][j][2])

            json_data[i][j]["distribution_a"] = collections.OrderedDict()
            json_data[i][j]["distribution_a"]["50010"] = str(prediction[3][i][j][0])
            json_data[i][j]["distribution_a"]["50011"] = str(prediction[3][i][j][1])
            json_data[i][j]["distribution_a"]["50012"] = str(prediction[3][i][j][2])

            json_data[i][j]["distribution_n"] = collections.OrderedDict()
            json_data[i][j]["distribution_n"]["50013"] = str(prediction[4][i][j][0])
            json_data[i][j]["distribution_n"]["50014"] = str(prediction[4][i][j][1])
            json_data[i][j]["distribution_n"]["50015"] = str(prediction[4][i][j][2])

    with open(output_path,'w') as out_file:
        json.dump(json_data, out_file, ensure_ascii=False, indent="\t")

    print(json.dumps(json_data, ensure_ascii=False, indent="\t"))
    return json.dumps(json_data, ensure_ascii=False, indent="\t")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str,
                help="input path")
    parser.add_argument('output_path', type=str,
                help="output path")

    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    hyperparam = Hparam.get_hparam()
    model = Model.Personality_Recognizer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        #Loading inputs
        #Morphological anayler
        komoran = Komoran()

        #Users can change the path of input
        input_data, sequence_length, mask  = load_data(input_path="Input/input.json", hyperparameter=hyperparam, analyzer=komoran)
        print("#Loading the model ...")
        saver.restore(sess, "Parameter_O/379")
        predict_o = sess.run(model.softmax_logits, feed_dict={model.input_x: input_data,
                                                        model.lstm_seq_length: sequence_length,
                                                        model.mask: mask,
                                                        model.dropout_keep_prob: hyperparam['test_dropout_keep_prob']})
        saver.restore(sess, "Parameter_C/251")
        predict_c = sess.run(model.softmax_logits, feed_dict={model.input_x: input_data,
                                                              model.lstm_seq_length: sequence_length,
                                                              model.mask: mask,
                                                              model.dropout_keep_prob: hyperparam[
                                                                  'test_dropout_keep_prob']})

        saver.restore(sess, "Parameter_E/288")
        predict_e = sess.run(model.softmax_logits, feed_dict={model.input_x: input_data,
                                                              model.lstm_seq_length: sequence_length,
                                                              model.mask: mask,
                                                              model.dropout_keep_prob: hyperparam[
                                                                  'test_dropout_keep_prob']})

        saver.restore(sess, "Parameter_A/388")
        predict_a = sess.run(model.softmax_logits, feed_dict={model.input_x: input_data,
                                                              model.lstm_seq_length: sequence_length,
                                                              model.mask: mask,
                                                              model.dropout_keep_prob: hyperparam[
                                                                  'test_dropout_keep_prob']})

        saver.restore(sess, "Parameter_N/282")
        predict_n = sess.run(model.softmax_logits, feed_dict={model.input_x: input_data,
                                                              model.lstm_seq_length: sequence_length,
                                                              model.mask: mask,
                                                              model.dropout_keep_prob: hyperparam[
                                                                  'test_dropout_keep_prob']})


        #writing and displaying results
        write_result(input_path=input_path, output_path=output_path, prediction=[predict_o, predict_c, predict_e, predict_a, predict_n])


"""

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
        all_results = []
        for data_type in ['A', 'C', 'E', 'N', 'O']:
            ckpt = tf.train.get_checkpoint_state(os.path.join('CNN-LSTM_{}'.format(data_type), 'Parameter'))
            saver.restore(sess, ckpt.model_checkpoint_path)
            predict = sess.run(tf.argmax(model.softmax_logits, -1), feed_dict={model.input_x: input_data,
                                                            model.lstm_seq_length: sequence_length,
                                                            model.mask: mask,
                                                            model.dropout_keep_prob: hyperparam['test_dropout_keep_prob']})

            results = []
            with open(input_path) as f:
                gold = json.load(f)
                for instances, predict_instances in zip(gold, predict):
                    predict_instances = predict_instances[:len(instances)]
                    result = []
                    for instance, predict_instance in zip(instances, predict_instances):
                        result_instance = {}
                        result_instance['predict'] = predict_instance
                        result_instance['utterance'] = instance['utterance']
                        result.append(result_instance)
                    results.append(result)
            all_results.append(results)


    for A_ins, C_ins, E_ins, N_ins, O_ins in zip(*all_results):
        for A_in, C_in, E_in, N_in, O_in in zip(A_ins, C_ins, E_ins, N_ins, O_ins):
            print(A_in['utterance'])
            print('pred-o: {}'.format(O_in['predict']))
            print('pred-c: {}'.format(C_in['predict']))
            print('pred-e: {}'.format(E_in['predict']))
            print('pred-a: {}'.format(A_in['predict']))
            print('pred-n: {}'.format(N_in['predict']))
        print('-'*100)
"""