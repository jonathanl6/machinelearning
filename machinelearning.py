import tensorflow as tf
import matplotlib.pyplot as plt

tf.reset_default_graph()  # resets graph
input_data = tf.placeholder(dtype=tf.float32, shape=None)  # insert placeholder
output_data = tf.placeholder(dtype=tf.float32, shape=None)

slope = tf.Variable(0.5, dtype=tf.float32)  # variables
intercept = tf.Variable(0.1, dtype=tf.float32)

model_operation = slope * input_data + intercept # the line

error = model_operation - output_data # operation squares the value testing computer
squared_error = tf.square(error)
loss = tf.reduce_mean(squared_error)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)  # minimizing loss
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()  # initializes global variables

x_values = [0, 1, 2, 3, 4]  # values
y_values = [0, 2, 4, 6, 8]
x2_values = [0, 1, 2, 3, 4]
y2_values = [8, 6, 4, 2, 0]
x3_values = [0, 1, 2, 3, 4]
y3_values = [4, 4, 4, 4, 4]


with tf.Session() as sess:  # runs operations
    sess.run(init)
    for i in range(2000):
        sess.run(train, feed_dict={input_data: x_values, output_data: y_values})
        if i % 100 == 0:
            print(sess.run([slope, intercept]))
            plt.plot(x_values, sess.run(model_operation, feed_dict={input_data: x_values}))  # plots

    print(sess.run(loss, feed_dict={input_data: x_values, output_data: y_values}))
    plt.plot(x_values, y_values, 'ro', 'Training Data')
    plt.plot(x_values, sess.run(model_operation, feed_dict={input_data: x_values}))

with tf.Session() as sess:  # runs operations
    sess.run(init)
    for i in range(2000):
        sess.run(train, feed_dict={input_data: x2_values, output_data: y2_values})
        if i % 100 == 0:
            print(sess.run([slope, intercept]))
            plt.plot(x2_values, sess.run(model_operation, feed_dict={input_data: x2_values}))  # plots

    print(sess.run(loss, feed_dict={input_data: x2_values, output_data: y2_values}))
    plt.plot(x2_values, y2_values, 'ro', 'Training Data')
    plt.plot(x2_values, sess.run(model_operation, feed_dict={input_data: x2_values}))

with tf.Session() as sess:  # runs operations
    sess.run(init)
    for i in range(2000):
        sess.run(train, feed_dict={input_data: x3_values, output_data: y3_values})
        if i % 100 == 0:
                print(sess.run([slope, intercept]))
                plt.plot(x3_values, sess.run(model_operation, feed_dict={input_data: x3_values}))  # plots

    print(sess.run(loss, feed_dict={input_data: x3_values, output_data: y3_values}))
    plt.plot(x3_values, y3_values, 'ro', 'Training Data')
    plt.plot(x3_values, sess.run(model_operation, feed_dict={input_data: x3_values}))

    plt.show()