def get_hparam():
    hparam = {'train_data_path':'../Eval_Gold/Data_N/train_N.json',
              'dev_data_path':'../Eval_Gold/Data_N/dev_N.json',
              'test_data_path':'../Eval_Gold/Data_N/test_N.json',
              'word_embedding_path':'../Eval_Gold/embedding_table.txt',
              'max_utterance_length':58,
              'max_dialogue_length':24,
              'embedding_dimension':300,
              'cnn_filter_size':[1, 2, 3],
              'cnn_filter_num':100,
              'lstm_hidden_dim':300,
              'train_dropout_keep_prob': 0.5,
              'test_dropout_keep_prob': 1.0,
              'batch_size':50,
              'epoch':15,
              'learning_rate':0.0005,
              'nb_classes':3}
    return hparam