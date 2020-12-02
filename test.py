import tensorflow as tf

input = tf.Variable(tf.random_normal([100, 28, 28, 1]))
filter = tf.Variable(tf.random_normal([5, 5, 1, 6]))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.initialize_all_variables())

op = tf.nn.conv2d(input, filter, strides = [1, 1, 1, 1], padding = 'VALID')
out = sess.run(op)
print(out)

#sudo docker run  --runtime=nvidia -it -p 8888:8888  tensorflow/tensorflow:1.15.2-gpu-py3-jupyter
