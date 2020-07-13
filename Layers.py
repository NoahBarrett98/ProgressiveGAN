import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Conv2D, LeakyReLU, AveragePooling2D
import numpy as np
# TODO: figure out how to actually remove layers from model so when we drop them, we do not continue to train them

# Weighted Sum #
class WeightedSum(tf.keras.layers.Add):
    """
    Merge layer, combines activations from two input layers
    such as two input paths in a discriminator or two output
    layers in a generator

    This is used during the growth phase of training when model
    is in transition from one image size to a new image size
    i.e 4x4 -> 8x8
    """
    # init with default value
    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        # tensor variable
        #self.alpha = tf.keras.backend.variable(alpha, name='ws_alpha')
        self.alpha = alpha

    def set_alpha(self, alpha):
        """
        set alpha for the layer
        :param alpha: float between 0.0-1.0
        :return:None
        """
        #self.alpha = tf.keras.backend.variable(alpha, name='ws_alpha')
        self.alpha = alpha

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)
        # merge two inputs with weight measured by alpha #
        # inputs[0] = old model, inputs[1] = new model
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output

# Minibatch Stdev Layer #
class MinibatchStdev(tf.keras.layers.Layer):
    """
    Only used in output block of the discriminator layer
    This layer provides a statistical summary of the batch of activations.
    The discriminator can learn to better detect batches of fake samples
    from batches of real samples. Therefore this layer encourages the generator
    (trained via discriminator) to create batches of samples with realistic
    batch statistics.
    """
    # initialize the layer
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # calculate the mean value for each pixel across channels
        mean = tf.keras.backend.mean(inputs, axis=0, keepdims=True)
        # calculate the squared differences between pixel values and mean
        squ_diffs = tf.keras.backend.square(inputs - mean)
        # calculate the average of the squared differences (variance)
        mean_sq_diff = tf.keras.backend.mean(squ_diffs, axis=0, keepdims=True)
        # add a small value to avoid a blow-up when we calculate stdev
        mean_sq_diff += 1e-8
        # square root of the variance (stdev)
        stdev = tf.keras.backend.sqrt(mean_sq_diff)
        # calculate the mean standard deviation across each pixel coord
        mean_pix = tf.keras.backend.mean(stdev, keepdims=True)
        # scale this up to be the size of one input feature map for each sample
        shape = tf.keras.backend.shape(inputs)
        # tiles Tensor by dimensions
        output = tf.keras.backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))
        # concatenate with the output
        combined = tf.keras.backend.concatenate([inputs, output], axis=-1)
        return combined

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        # create a copy of the input shape as a list
        input_shape = list(input_shape)
        # add one to the channel dimension (assume channels-last)
        input_shape[-1] += 1
        # convert list to a tuple
        return tuple(input_shape)

# Pixel Normalization #
class PixelNormalization(tf.keras.layers.Layer):
    """
    The generator and discriminator in Progressive growing GAN differs from
    most as it does not use Batch Normalization. instead each pixel in activation
    maps are normalized to unit length. this is known as pixelwise feature vector
    normalization. Normalization is only usd in the generator.

    To disallow the scenario where the magnitudes in the generator and discriminator
    spiral out of control as a result of competition
    """
    # initialize the layer
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    # perform the operation
    def call(self, inputs):
        # calculate square pixel values
        values = inputs ** 2.0
        # calculate the mean pixel values
        mean_values = tf.keras.backend.mean(values, axis=-1, keepdims=True)
        # ensure the mean is not zero
        mean_values += 1.0e-8
        # calculate the sqrt of the mean squared value (L2 norm)
        l2 = tf.keras.backend.sqrt(mean_values)
        # normalize values by the l2 norm
        normalized = inputs / l2
        return normalized

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        return input_shape

# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
    """
    using wasserstein loss to simplify implementation
    :param y_true: groundtruth img
    :param y_pred: prediction img
    :return: wasserstein loss
    """
    return tf.keras.backend.mean(y_true * y_pred)

class gen_block(tf.keras.layers.Layer):
    """
    each block is concerned with two things the output shape and
    number of filters

    --upsample will double our output dims every block.
    """
    def __init__(self, num_filters, reduce_filters, upsample=True, is_end=True, **kwargs):
        super(gen_block, self).__init__(**kwargs)
        # bool to remove upsamples where neccesary
        self.upsample = upsample

        # on creation it will be end.
        self.is_end = is_end

        # after 32x32 must start reducing filter size
        if reduce_filters:
            self.num_filters = int(num_filters / 2)
        else:
            self.num_filters = num_filters

        self.upspl1 = UpSampling2D()
        self.conv1 = Conv2D(filters=self.num_filters,
                  kernel_size=(3,3),padding="same")
        self.act1 = LeakyReLU(alpha=0.2)
        self.conv2 = Conv2D(filters=self.num_filters,
                   kernel_size=(3, 3), padding="same")
        self.act2 = LeakyReLU(alpha=0.2)

        # for if last
        self.conv_last1 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')
        self.act_last1 = LeakyReLU(alpha=0.2)
        self.conv_last2 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')
        self.act_last2 = LeakyReLU(alpha=0.2)
        self.RGB_out = Conv2D(3, (1, 1), padding='same', kernel_initializer='he_normal')

    def deactivate_output(self):
        """
        This ensures that we are not training the outputs once this output layer essentially
        deprecates
        """
        self.conv_last1.trainable = False
        self.act_last1.trainable = False
        self.conv_last2.trainable = False
        self.act_last2.trainable = False
        self.RGB_out.trainable = False

    def call(self, inputs):
        x= inputs
        if self.upsample:
            x = self.upspl1(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)

        if self.is_end:
            x = self.conv_last1(x)
            x = self.act_last1(x)
            x = self.conv_last2(x)
            x = self.act_last2(x)
            x = self.RGB_out(x)

        return x


class dis_block(tf.keras.layers.Layer):
    """
    each block is concerned with two things the output shape and
    number of filters

    --downsample will halve our output dims every block.
    """
    def __init__(self, num_filters, decrease_filters, is_top=True, **kwargs):
        super(dis_block, self).__init__(**kwargs)
        # if is top, will include the input layer for it
        self.is_top = is_top
        self.num_filters = num_filters

        # filters increase after first conv layer
        if decrease_filters:
            self.num_filters = int(self.num_filters / 2)

        # input to be used when instance is the top of the model.
        self.input_conv = Conv2D(self.num_filters, (1, 1), padding='same', kernel_initializer='he_normal')
        self.input_act = LeakyReLU(alpha=0.2)
        # until 32x32 must double filter size


        self.conv1 = Conv2D(filters=self.num_filters,
                  kernel_size=(3,3),padding="same")
        self.act1 = LeakyReLU(alpha=0.2)

        # must scale back up
        if decrease_filters:
            self.num_filters = int(self.num_filters * 2)

        self.conv2 = Conv2D(filters=self.num_filters,
                   kernel_size=(3, 3), padding="same")
        self.act2 = LeakyReLU(alpha=0.2)

        # uses average pooling for downsample
        self.dnspl1 = AveragePooling2D()

    def deactivate_input(self):
        """
        once old pass in
        """
        self.input_conv.trainable = False
        self.input_act.trainable = False

    def call(self, inputs):
        x= inputs
        if self.is_top:
            x = self.input_conv(x)
            x = self.input_act(x)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.dnspl1(x)

        return x
