
from tensorflow.examples.tutorials.mnist import input_data
path1 = 'F:\\OneDrive\\mytensorflow\\MNIST_data'
path2 = '~/home/zhouying/mytensorflow/MNIST_data'
mnist = input_data.read_data_sets(path,one_hot=True)

import tensorflow as tf
import numpy as np

learning_rate = 0.0001
epochs = 10
batch_size = 128
display_step = 100

# networks parameters
n_input = 28
n_steps = 28
n_hidden = 128
n_classes = 10

#define the graph
x = tf.placeholder('float32',[None,n_steps,n_input])
y = tf.placeholder('float32',[None,n_classes])

weights = {
	'hidden':tf.Variable(tf.random_normal([n_input,n_hidden])),
	'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))
}

bias = {
	'hidden':tf.Variable(tf.random_normal([n_hidden])),
	'out':tf.Variable(tf.random_normal([n_classes]))
}

lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden,forget_bias=0.0,state_is_tuple=True)
_state = lstm_cell.zero_state(batch_size,tf.float32)

a1 = tf.transpose(x,[1,0,2])
a2 = tf.reshape(a1,[-1,n_input])
a3 = tf.matmul(a2,weights['hidden']+bias['hidden'])
a4 = tf.split(a3,n_steps,0)

outputs,states = tf.nn.static_rnn(lstm_cell,a4,initial_state=_state)
a5 = tf.matmul(outputs[-1],weights['out'])+bias['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=a5,labels=y))
#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(a5,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(epochs):
	batch_xs,batch_ys = mnist.train.next_batch(batch_size)
	batch_xs = batch_xs.reshape((batch_size,n_steps,n_input))
	sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys})
	if step%display_step == 0:
		loss,acc = sess.run([cost,accuracy],feed_dict={x:batch_xs,y:batch_ys})
		print('loss %f,accuracy: %f',loss,acc)

