import Model, Util, Hparam
import tensorflow as tf
from tqdm import tqdm
import collections

if __name__=="__main__":
    #Once target language is specified here, you don't need to care of the type of language anymore during training or inference.
    hyperparam = Hparam.get_hparam()

    #Loading the model
    model = Model.Personality_Recognizer()

    # Tensorboard_setting
    summary = tf.summary.merge_all()
    global_step = 0

    # Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        #Strating to train model
        print("#Training CNN_LSTM model:")
        print("- Step1) Loading train/dev/test dataset loading ...")
        #Load real data
        train_x, train_y, train_sequence_length, train_mask, dev_x, dev_y, dev_sequence_length, dev_mask, test_x, test_y, test_sequence_length, test_mask = Util.load_data(
            hyperparam)

        train_x_batch, train_y_batch, train_seq_length_batch, train_mask_batch = Util.load_batches(train_x,
                                                                                                   train_y,
                                                                                                   train_sequence_length,
                                                                                                   train_mask,
                                                                                                   hyperparam)

        #Initializing variables
        init = tf.global_variables_initializer()
        sess.run(init)

        #Tensorboard writer
        writer_train = tf.summary.FileWriter("log/train")
        writer_dev = tf.summary.FileWriter("log/dev")
        writer_train.add_graph(sess.graph)
        writer_dev.add_graph(sess.graph)

        train_cost = 0
        dev_cost = 0

        train_acc = 0
        dev_acc = 0

        valid_acc = 0

        print("Step 2) Training model ...")
        for epoch_num in range(0, hyperparam["epoch"]):
            print(epoch_num, "Epoch")
            batch_size = len(train_x_batch)
            for batch_num in tqdm(range(0, batch_size)):
                train_sum, _, train_acc, train_cost = sess.run([summary, model.optimizer, model.accuracy, model.loss],
                                                                feed_dict={model.input_x: train_x_batch[batch_num],
                                                                           model.output_y: train_y_batch[batch_num],
                                                                           model.lstm_seq_length: train_seq_length_batch[batch_num],
                                                                           model.mask: train_mask_batch[batch_num],
                                                                           model.dropout_keep_prob: hyperparam['train_dropout_keep_prob']})

                writer_train.add_summary(train_sum, global_step=global_step)

                dev_sum, dev_acc, dev_cost = sess.run([summary, model.accuracy, model.loss],
                                                       feed_dict={model.input_x: dev_x,
                                                                  model.output_y: dev_y,
                                                                  model.lstm_seq_length: dev_sequence_length,
                                                                  model.mask: dev_mask,
                                                                  model.dropout_keep_prob: hyperparam[
                                                                      'test_dropout_keep_prob']})
                writer_dev.add_summary(dev_sum, global_step=global_step)
                global_step += 1

                if global_step > 250 and global_step < 420:
                    if valid_acc < dev_acc:
                        print("\n-", global_step, "Step")
                        valid_acc = dev_acc
                        saver.save(sess, "Parameter/" + str(global_step))
                        test_x_batch, test_y_batch, test_seq_length_batch, test_mask_batch = Util.load_batches_test(test_x,
                                                                                                                    test_y,
                                                                                                                    test_sequence_length,
                                                                                                                    test_mask,
                                                                                                                    hyperparam)

                        # Test Result
                        class_correct = collections.OrderedDict({0: 0, 1: 0, 2: 0})
                        class_total = collections.OrderedDict({0: 0, 1: 0, 2: 0})

                        test_batch_size = len(test_x_batch)
                        for test_batch_num in range(0, test_batch_size):
                            t_x, t_y = sess.run([model.prediction , model.output_y],
                                                                   feed_dict={model.input_x: test_x_batch[test_batch_num],
                                                                              model.output_y: test_y_batch[test_batch_num],
                                                                              model.lstm_seq_length: test_seq_length_batch[test_batch_num],
                                                                              model.mask: test_mask_batch[test_batch_num],
                                                                              model.dropout_keep_prob: hyperparam[
                                                                                  'test_dropout_keep_prob']})
                            for ids in range(len(t_x)):
                                counted = test_seq_length_batch[test_batch_num][ids]
                                for i in range(counted):
                                    class_total[t_y[ids][i]] += 1
                                    if t_y[ids][i] == t_x[ids][i]:
                                        class_correct[t_x[ids][i]] += 1

                        print("==================================")
                        print("Result: accuracy")
                        total_count = 0
                        correct_count = 0
                        for k,v in class_total.items():
                            total_count += class_total[k]
                        for k,v in class_correct.items():
                            correct_count += class_correct[k]
                        print(correct_count/total_count, "(", correct_count,"/",total_count,")")


            #Summary for each epoch
            print("#Epoch:", str(epoch_num))
            print(" - Train/Dev Cost:", train_cost, "/", dev_cost)
            print(" - Train/Dev Accuracy:", train_acc, "/", dev_acc)

