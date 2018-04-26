import numpy as np
import tensorflow as tf
import cv2
from newcnn3 import *
import time

'''
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

image,label = data.test.next_batch(60)
for each in range(0,len(image)):
    cv2.imshow("test data",cv2.resize(image[each],(28,28)))
    print(label[each])
    cv2.waitKey()
    

'''
##savingpath = "/tmp/convocheck"
##savingpath = "/tmp/emotion/conweights"
##savingpath = "/tmp/emotion_nor/conweights" 2 layer working
savingpath = "/tmp/emotion_nor3/conweights"
session = tf.Session()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()

saver.restore(sess=session, save_path=savingpath)
##image,label = data.test.next_batch(60)
#print(image)


test_data = np.load(r"C:\Users\acer\Desktop\emotion detection\train\test_data1.npy")
test_label = np.load(r"C:\Users\acer\Desktop\emotion detection\train\test_label1.npy")




#input

feed_dict = {x: test_data, y_true: test_label}

# Calculate the predicted class using TensorFlow.
cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)
print("acuracy:",session.run(accuracy, feed_dict=feed_dict)*100,"%")

