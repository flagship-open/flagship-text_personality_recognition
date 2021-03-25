from flask import Flask, session, g, request, render_template, redirect
import os
import sys
import tensorflow as tf
import json
import collections
from tflearn.data_utils import pad_sequences
from konlpy.tag import Komoran
import Model, Util, Hparam
from Infer import load_data, write_result

import time


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

base_dir = os.path.abspath(os.path.dirname(__file__) + '/')
sys.path.append(base_dir)
app = Flask(__name__)

class Handler:
    def __init__(self):
        self.hyperparam = Hparam.get_hparam()
       

        print("#Loading all models ...")
        self.komoran = Komoran()
        self.model = Model.Personality_Recognizer()
        
        self.saver_O = tf.train.Saver()
        self.sess_O = tf.Session()
        self.saver_O.restore(self.sess_O, "CNN-LSTM_O/Parameter/251")

        self.saver_C = tf.train.Saver()
        self.sess_C = tf.Session()
        self.saver_C.restore(self.sess_C, "CNN-LSTM_C/Parameter/251")

        self.saver_E = tf.train.Saver()
        self.sess_E = tf.Session()
        self.saver_E.restore(self.sess_E, "CNN-LSTM_E/Parameter/251")

        self.saver_A = tf.train.Saver()
        self.sess_A = tf.Session()
        self.saver_A.restore(self.sess_A, "CNN-LSTM_A/Parameter/388")

        self.saver_N = tf.train.Saver()
        self.sess_N = tf.Session()
        self.saver_N.restore(self.sess_N, "CNN-LSTM_N/Parameter/251")







    def predict(self, input_path, output_path):
        input_data, sequence_length, mask = load_data(input_path=input_path, hyperparameter=self.hyperparam,
                                                          analyzer=self.komoran)
        start = time.time()
        predict_o = self.sess_O.run(self.model.softmax_logits, feed_dict={self.model.input_x: input_data,
                                                                            self.model.lstm_seq_length: sequence_length,
                                                                            self.model.mask: mask,
                                                                            self.model.dropout_keep_prob: self.hyperparam[
                                                                      'test_dropout_keep_prob']})
            
        predict_c = self.sess_C.run(self.model.softmax_logits, feed_dict={self.model.input_x: input_data,
                                                                            self.model.lstm_seq_length: sequence_length,
                                                                            self.model.mask: mask,
                                                                            self.model.dropout_keep_prob: self.hyperparam[
                                                                      'test_dropout_keep_prob']})

        predict_e = self.sess_E.run(self.model.softmax_logits, feed_dict={self.model.input_x: input_data,
                                                                            self.model.lstm_seq_length: sequence_length,
                                                                            self.model.mask: mask,
                                                                            self.model.dropout_keep_prob: self.hyperparam[
                                                                      'test_dropout_keep_prob']})


        predict_a = self.sess_A.run(self.model.softmax_logits, feed_dict={self.model.input_x: input_data,
                                                                            self.model.lstm_seq_length: sequence_length,
                                                                            self.model.mask: mask,
                                                                            self.model.dropout_keep_prob: self.hyperparam[
                                                                      'test_dropout_keep_prob']})


        predict_n = self.sess_N.run(self.model.softmax_logits, feed_dict={self.model.input_x: input_data,
                                                                            self.model.lstm_seq_length: sequence_length,
                                                                            self.model.mask: mask,
                                                                            self.model.dropout_keep_prob: self.hyperparam[
                                                                      'test_dropout_keep_prob']})

        print("time :", time.time() - start)
        # writing and displaying results
        return write_result(input_path=input_path, output_path=output_path, prediction=[predict_o, predict_c, predict_e, predict_a, predict_n])


@app.route("/predict", methods=["POST"])
def index():
    global handler
    data = request.get_json()
    input_path = data.get('input_path', '[[{"speaker": "A","utterance": "로리 새 프로젝트를 시작하는데 로리씨가 도와줬으면 좋겠어."}]]')
    output_path = data.get('output_path', 'Output/output.json')

    return {'result': handler.predict(input_path=input_path, output_path=output_path)}


if __name__ == "__main__":
    handler = Handler()

    FLASK_DEBUG = os.getenv('FLASK_DEBUG', False)
    
    app.run(host="0.0.0.0", debug=True, port=8080, threaded=True)
