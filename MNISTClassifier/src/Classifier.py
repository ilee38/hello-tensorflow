'''
Created on May 3, 2017

@author: iramlee
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

#nodes for imput images (x) and target classes (y)
'''
Here we assign it a shape of [None, 784], where 784 is the 
dimensionality of a single flattened 28 by 28 pixel MNIST 
image, and None indicates that the first dimension, 
corresponding to the batch size, can be of any size.
'''
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#Model parameters as Variables.
'''
In machine learning applications, one generally has the model parameters be Variables.
'''
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())     #Initialize all Variables

#Implement the regression Model
y = tf.matmul(x,W) + b

#Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#Optimizer (Gradient Descent)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Training loop
for _ in range(1000):
    batch = mnist.train.next_batch(100)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  
#Model evaluation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
'''
That gives us a list of booleans. To determine what fraction are correct, 
we cast to floating point numbers and then take the mean. For example, 
[True, False, True, True] would become [1,0,1,1] which would become 0.75.
'''
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

