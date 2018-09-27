from helpers import *

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

#Colour Channles
num_channels = 1

num_classes = 10

def CNN(x_image):

    layer_conv1, weights_1 = new_conv_layer(input=x_image,
                                    num_input_channels=num_channels,
                                    filter_size = filter_size1,
                                    num_filters = num_filters1,
                                    use_pooling= True)

    layer_conv2, weights_2 = new_conv_layer(input=layer_conv1,
                                    num_input_channels=num_filters1,
                                    filter_size = filter_size2,
                                    num_filters = num_filters2,
                                    use_pooling= True)

    flat_layer, num_features = flatten_layer(layer_conv2)


    fc1 = new_fc_layer(flat_layer,
                    num_features,
                    num_outputs=fc_size,)

    fc2_logits = new_fc_layer(fc1,
                num_inputs=fc_size,
                num_outputs=num_classes,
                use_relu=False)

    return fc2_logits