import tensorflow as tf

# Read the dataset
mnist = tf.contrib.learn.datasets.load_dataset('mnist')
#tf.examples.tutorials.mnist.input_data.read_data_sets("MNIST_data/")

# Since each image is 28*28 pixels, we flatten it into 1*784
# x will be a n*784 matrix, where n is the batch size (decided later, hence None now)
x = tf.placeholder(tf.float32, [None, 784])
# y_ is the answer label, n*1, each row contains the mnist answers (0, 1, ..., 9)
y_ = tf.placeholder(tf.int32, [None])

# Use one-hot to change [3] into [0, 0, 0, 1, 0, 0, 0, 0, 0, 0], hence y_one is n*10
y_one = tf.one_hot(y_, 10)

# ======================== #
# layer1 weights and bias, w1 is a 784*512 matrix, initializer should have a default choice
w1 = tf.get_variable('w1', shape=[784, 512], initializer=tf.glorot_uniform_initializer())
# b1 should be 1*512? why not shape=[1, 512] here?
b1 = tf.get_variable('b1', shape=[512], initializer=tf.zeros_initializer)

# layer2 weights and bias
w2 = tf.get_variable('w2', shape=[512, 10], initializer=tf.glorot_uniform_initializer())
b2 = tf.get_variable('b2', shape=[10], initializer=tf.zeros_initializer)

# n*512 matrix
h1 = tf.matmul(x, w1) + b1
# apply activation
h1_relu = tf.nn.relu(h1)
# n*10 matrix
y = tf.matmul(h1_relu, w2) + b2
# ======================== #

# The section above can be written in the high-level api:
# 512 stands for 512 nodes in this layer
h1_relu = tf.layers.dense(x, 512, activation=tf.nn.relu)
y = tf.layers.dense(h1_relu, 10)
# ======================== #

# Apply softmax on y to make sum = 1 and compute its cross entropy (error) with y_one
# Use reduce_mean to average the batch values into a single value
# 1*1 matrix
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_one, logits=y))

# (optional) compute the accuracy
# argmax find the column with largest value for each row
# 1 is the dimension, which indicates using column to find
# n*1 matrix
y_pred = tf.argmax(y, 1, output_type=tf.int32)
# compare it with y_, if hit, 1, else 0
# n*1 matrix
correct_prediction = tf.equal(y_pred, y_)
# Get average, 1*1 matrix
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# ======================== #

# Apply an optimizer on the error,
# which will trace to find the trainable variables and apply derivative on them when run.
# All the x, y_, y_one, y, y_pred, h1_relu, cross_entropy, ...etc are Tensors,
# you can put any of them into minimize(), but make sure you know what you're doing.
# train_step will be an operation
train_step = tf.train.AdamOptimizer(0.05).minimize(cross_entropy)

# Create a session for running
sess = tf.Session()
# Initialize all the variables above?
sess.run(tf.global_variables_initializer())

# testing data
test_xs = mnist.test.images
test_ys = mnist.test.labels

# train 1000 steps
for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(128)
    # trigger the operations/tensors to get the values (along with variable updates)
    _, ce, acc = sess.run([train_step, cross_entropy, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
    if (step % 100 == 0):
        acc_test = sess.run(accuracy, feed_dict={x: test_xs, y_: test_ys})
        print("Accuracy - Train: %.4f, Test: %.4f" % (acc, acc_test))
