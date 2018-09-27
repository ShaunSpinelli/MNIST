import tensorflow as tf
from convnet import CNN

# The number of pixels in each dimension of an image.
img_size = 28#data.img_size

# The images are stored in one-dimensional arrays of this length.
img_size_flat = 784 #data.img_size_flat

# Tuple with height and width of images used to reshape arrays.
img_shape = (28,28)#data.img_shape

# Number of classes, one class for each of 10 digits.
num_classes = 10 #data.num_classes


x = tf.placeholder(tf.float32, [None, img_size_flat], name="x")

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, [None, num_classes], name="y_true")

y_labels = tf.argmax(y_true, axis=1)

logits = CNN(x_image)

