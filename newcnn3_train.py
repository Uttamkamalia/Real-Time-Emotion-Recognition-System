import numpy as np
import tensorflow as tf
from newcnn3 import *
import time

#load train data
##from tensorflow.examples.tutorials.mnist import input_data
#data = input_data.read_data_sets('data/MNIST/', one_hot=True)
##for line data##train_features = np.load(r"C:\Users\acer\Desktop\emotion detection\train\train_features1.npy")
train_features = np.load(r"C:\Users\acer\Desktop\emotion detection\train\train_contrastfeatures.npy") ## size(35887, 48, 48, 1)
train_onehot = np.load(r"C:\Users\acer\Desktop\emotion detection\train\new_train_labels1.npy")
train_labels = np.load(r"C:\Users\acer\Desktop\emotion detection\train\not_one_hot_labels1.npy")
print(train_onehot.shape)

# total size of train set = 24796

totalsize = 24796
# batch size for training
train_batch_size = 50
##savingpath = "/tmp/convocheck"
##savingpath = "/tmp/emotion/conweights" digit recognizer
#savingpath = "/tmp/emotion_nor/conweights"  2 layer working
savingpath = "/tmp/emotion_nor3/conweights"



# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations,load=False):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    #load trained weights
    if load:
        saver.restore(sess=session, save_path=savingpath)
    

    # Start-time used for printing time-usage below.
    start_time = time.time()
    t=0
    for i in range(total_iterations,
                   total_iterations + num_iterations):
        print("iteration:",i)

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        ##x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        x_batch = train_features[t:t+50]
        y_true_batch = train_onehot[t:t+50]
        t+= 50
        t = t%totalsize

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))
            saver.save(sess=session, save_path=savingpath)

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
   # print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    



# start training::: 495
print("training started")
optimize(10000,load=False)
print("optimization done")





