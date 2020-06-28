import tensorflow as tf

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
        self.alpha = tf.keras.backend.variable(alpha, name='ws_alpha')

    def set_alpha(self, alpha):
        """
        set alpha for the layer
        :param alpha: float between 0.0-1.0
        :return:None
        """
        self.alpha = tf.keras.backend.variable(alpha, name='ws_alpha')

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