import tensorflow as tf
import numpy as np
import Hparam, Util
class Personality_Recognizer(object):
    def __init__(self):
        #Setting hyperparameters
        self.hparam = Hparam.get_hparam()
        #Placeholder for input, output in a batch
        self.input_x = tf.placeholder(tf.int32, [None, self.hparam['max_utterance_length']], name="input_x")
        self.output_y = tf.placeholder(tf.int64, [None, self.hparam['max_dialogue_length']], name="output_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        #First layer: embedding layer for network
         #Loading vocabulary, then producing embedded input
        self.ids2word, self.word2ids, self.ids2vec = Util.load_vocab(self.hparam['word_embedding_path'])
         #Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("Embedding_Layer"):
            self.network_word_embedding = np.zeros([len(self.ids2word), self.hparam['embedding_dimension']])
            for ids, vector in self.ids2vec.items():
                self.network_word_embedding[ids] = np.fromstring(vector, dtype=np.float32, sep=' ')
            self.network_word_embeddings = tf.Variable(self.network_word_embedding, trainable=True, dtype=tf.float32)
            self.embedded_inputs = tf.nn.embedding_lookup(self.network_word_embeddings, self.input_x)
            self.expanded_embeddded_inputs = tf.expand_dims(self.embedded_inputs, -1)
        # Second layer: convolution with max-pool layer for each filter size -> producing sentence representations
        self.pooled_outputs = []
        for index, filter_size in enumerate(self.hparam['cnn_filter_size']):
            with tf.name_scope("CONV_with_MAXPOOL_Filter_%s" % filter_size):
                # Convolution Layer
                self.filter_shape = [filter_size, self.hparam['embedding_dimension'], 1, self.hparam['cnn_filter_num']]
                self.W = tf.Variable(tf.truncated_normal(self.filter_shape, stddev=0.1), name="W")
                self.b = tf.Variable(tf.constant(0.1, shape=[self.hparam['cnn_filter_num']]), name="b")
                self.convolution = tf.nn.conv2d(self.expanded_embeddded_inputs, self.W, strides=[1, 1, 1, 1], padding="VALID", name="convolution")
                # Applying nonlinearity
                self.convolution_with_relu = tf.nn.relu(tf.nn.bias_add(self.convolution, self.b), name="convolution_with_relu")
                # Maxpooling over the outputs
                self.pooled = tf.nn.max_pool(self.convolution_with_relu,
                                             ksize=[1, self.hparam['max_utterance_length'] - filter_size + 1, 1, 1],
                                             strides=[1, 1, 1, 1],
                                             padding='VALID',
                                             name="pool")
                self.pooled_outputs.append(self.pooled)
        self.total_num_filters = self.hparam['cnn_filter_num'] * len(self.hparam['cnn_filter_size'])
        self.pooled_representation = tf.concat(self.pooled_outputs, 3)
        # Sentence Representation: [batch, representation_dimension]
        self.sentence_representations = tf.reshape(self.pooled_representation, [-1, self.total_num_filters],name="sentence_representations")
        self.reshaped_sentence_representations = tf.reshape(self.sentence_representations,
                                                        [-1, self.hparam['max_dialogue_length'],
                                                         len(self.hparam['cnn_filter_size']) * self.hparam['cnn_filter_num']])
        #Third layer: LSTM at dialogue level
        #LSTM
        with tf.name_scope("LSTM"):
            self.cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.hparam['lstm_hidden_dim'], state_is_tuple=True)
            self.dropout_cell = tf.contrib.rnn.DropoutWrapper(cell=self.cell, output_keep_prob=self.dropout_keep_prob)
            self.initial_state = self.cell.zero_state(tf.shape(self.reshaped_sentence_representations)[0], tf.float32)
            self.lstm_seq_length = tf.placeholder(tf.int32, [None])
            self.lstm_outputs, self.lstm_states = tf.nn.dynamic_rnn(self.dropout_cell, inputs=self.reshaped_sentence_representations,
                                                                    sequence_length=self.lstm_seq_length,
                                                                    initial_state=self.initial_state,
                                                                    dtype=tf.float32)
        #Firth layer: FNN
        with tf.name_scope("FNN"):
            self.lstm_outputs_drop = tf.nn.dropout(self.lstm_outputs, self.dropout_keep_prob)
            self.FNN_logits = tf.layers.dense(self.lstm_outputs_drop, units=self.hparam['nb_classes'],
                                              kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.softmax_logits = tf.nn.softmax(self.FNN_logits)

        #Fifth layer: Mean cross-entropy loss
        with tf.name_scope("Loss"):
            self.mask = tf.placeholder(dtype=tf.float32,shape=[None, self.hparam['max_dialogue_length']])
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output_y, logits=self.FNN_logits) * self.mask
            self.loss = tf.reduce_mean(tf.reduce_sum(self.cross_entropy, 1) / tf.reduce_sum(self.mask, 1))
            #To be recorded in Tensorboard
            tf.summary.scalar("loss_summary", self.loss)
        self.optimizer = tf.train.AdamOptimizer(self.hparam['learning_rate']).minimize(self.loss)

        #Seventh layer: Prediction
        with tf.name_scope("Accuracy"):
            self.prediction = tf.argmax(self.FNN_logits, -1)
            self.casting = tf.cast(tf.equal(self.prediction, self.output_y), tf.float32) * self.mask
            self.accuracy = tf.reduce_sum(self.casting) / tf.reduce_sum(self.mask)
            #To be recorded in Tensorboard
            tf.summary.scalar("accuracy_summary", self.accuracy)