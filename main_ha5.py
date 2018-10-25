import numpy as np
import scipy
import matplotlib.pyplot as plt
from keras.datasets import mnist
from util import func_confusion_matrix
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import util
from sklearn import preprocessing


# load (downloaded if needed) the MNIST dataset
(x_train1, y_train1), (x_test, y_test) = mnist.load_data()


# transform each image from 28 by28 to a 784 pixel vector
pixel_count = x_train1.shape[1] * x_train1.shape[2]
x_train1 = x_train1.reshape(x_train1.shape[0], pixel_count).astype('float32')
x_test = x_test.reshape(x_test.shape[0], pixel_count).astype('float32')

# normalize inputs from gray scale of 0-255 to values between 0-1
x_train1 = x_train1 / 255
x_test = x_test / 255

# Please write your own codes in responses to the homework assignment 5

validation = 10000;

x_validate = x_train1[validation:,]
y_validate = y_train1[validation:,]

x_train = x_train1[validation:len(x_train1),0:784]
y_train = y_train1[validation:len(y_train1)]

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))


#Method #1 using GradientDescent

#sess.run(tf.global_variables_initializer())


y= tf.matmul(x,W) + b
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#for _ in range(1000):
#  batch = mnist.train.next_batch(100)
#  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


keys = {0:[1,0,0,0,0,0,0,0,0,0],1:[0,1,0,0,0,0,0,0,0,0],2:[0,0,1,0,0,0,0,0,0,0],3:[0,0,0,1,0,0,0,0,0,0],4:[0,0,0,0,1,0,0,0,0,0],5:[0,0,0,0,0,1,0,0,0,0],6:[0,0,0,0,0,0,1,0,0,0],7:[0,0,0,0,0,0,0,1,0,0],8:[0,0,0,0,0,0,0,0,1,0],9:[0,0,0,0,0,0,0,0,0,1]}

ALPHA = 0.01
TRAIN_STEPS = 2500

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


#First Convolutional Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second Convolutional layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_conv2 = tf.tanh(conv2d(h_pool1, W_conv2) + b_conv2)

h_pool2 = max_pool_2x2(h_conv2)


#Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1 = tf.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

def returnLabel(data):
    labels = {}
    for i in range(0,len(data)):
        labels[i] = keys[data[i]]
    return list(labels.values())

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(ALPHA).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  vBatch_y = returnLabel(y_validate)
  iterSize = 100
  for i in range(500):
    batch_x= x_train[(iterSize*i):(iterSize*i)+iterSize,:]
    batch_y= y_train[(iterSize*i):(iterSize*i)+iterSize]
        #return (batch_y)
    batch_y= list(returnLabel(batch_y))
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: x_validate, y_: vBatch_y, keep_prob: 1.0})
      #train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})
    if train_accuracy >= 0.99:
        break
    
  y_test_copy = y_test 
  y_test_keyed= returnLabel(y_test)
  predicted_output = sess.run(tf.argmax(y_conv,1), feed_dict={x: x_test, y_: y_test_keyed, keep_prob: 1.0})
  print('test accuracy %g' % accuracy.eval(feed_dict={x: x_test, y_: y_test_keyed, keep_prob: 1.0}))
 
  def func_calConfusionMatrix(predY, trueY):
    accuracy_rate = 0.00
    accuracy_sum = 0
    l1 = preprocessing.LabelEncoder()
    holdTrueLabel =l1.fit_transform(trueY)
    print(holdTrueLabel)
    l2 = preprocessing.LabelEncoder()
    holdPredLabel = l2.fit_transform(predY)
    matrix_size=(max(holdTrueLabel)+1)
    cf_matrix=np.zeros((matrix_size,matrix_size))    
     #return cf_matrix
    for i in range(0,len(holdTrueLabel)):
        cf_matrix[holdTrueLabel[i],holdPredLabel[i]] += 1    
    for i in range(0,matrix_size):
        for j in range(0,matrix_size):
            if i == j:
                accuracy_sum += cf_matrix[i][j]                
    accuracy_rate = accuracy_sum/len(trueY)        
    recall_list = cf_matrix.sum(axis=1)         
    for i in range(0,matrix_size):
        for j in range(0,matrix_size):
            if i == j:
                recall_list[i] = cf_matrix[i][j]/recall_list[i]                
    precision_list = cf_matrix.sum(axis=0)    
    for i in range(0, matrix_size):
        for j in range(0,matrix_size):
            if i== j:
                precision_list[i]= cf_matrix[i][j]/precision_list[i]      
    print("matrix size",matrix_size)   
    print(cf_matrix)    
    print("accuracy_rate  ",accuracy_rate)
    print("recall_array", recall_list) 
    print("precision_list", precision_list)     
   
    return accuracy_rate, recall_list, precision_list  
  
  
accuracy_rate, recall_list, precision_list= func_calConfusionMatrix(predicted_output, y_test_copy)
  
#print(conf_matrix)
#print(accuracy_rate)
#print(recall_list)
#print(precision_list)
    
  
