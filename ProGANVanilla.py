"""
Progressive Growing Gan
The Progressive Growing GAN is an extension to the GAN training procedure that
involves training a GAN to generate very small images, such as 4×4,
and incrementally increasing the size of the generated images to 8×8, 16×16,
until the desired output size is met. This has allowed the progressive GAN to generate
photorealistic synthetic faces with 1024×1024 pixel resolution.

described in the 2017 paper by Tero Karras, et al. from Nvidia
titled “Progressive Growing of GANs for Improved Quality, Stability, and Variation.”
"""
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU
from Layers import *


# TODO: transition from batch normalization in both gen and dis
class prog_model(Model):
    def __init__(self, **kwargs):
        super(prog_model, self).__init__(**kwargs)
        self._weighted_sum_alpha = 0.0

    def set_alpha(self, alpha):
        """
        update alpha in all weighted sums
        :param alpha: float between 0.0-1.0
        :return: None
        """
        assert alpha <= 1.0 and alpha >= 0.0
        # update base #
        for b_layer in self.layers[0].layers:
            if isinstance(b_layer, WeightedSum):
                b_layer.set_alpha(alpha)

        # update current #
        for c_layer in self.layers[1].layers:
            if isinstance(c_layer, WeightedSum):
                c_layer.set_alpha(alpha)

        # update fadein #
        for f_layer in self.layers[2].layers:
            if isinstance(f_layer, WeightedSum):
                f_layer.set_alpha(alpha)

    def call(self, input, fadein):
        if fadein:
            return self._fadein_model(input)
        else:
            return self._current_model(input)

class Prog_Discriminator(prog_model):
    def __init__(self,
                 leakyrelu_alpha=0.2,
                 init_lr=0.001,
                 init_beta_1=0,
                 init_beta_2=0.99,
                 init_epsilon=10e-8,
                 **kwargs
                 ):
        # call the parent constructor
        super(Prog_Discriminator, self).__init__(**kwargs)
        self.leakyrelu_alpha = leakyrelu_alpha
        ### Construct base model ###
        # input starts as 4x4x3
        x = Input(shape=(4,4,3))
        # conv 1x1
        x_prime = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(x)
        x_prime = LeakyReLU(alpha=leakyrelu_alpha)(x_prime)
        # conv 3x3 (output block)
        x_prime = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x_prime)
        x_prime = tf.keras.layers.BatchNormalization()(x_prime)
        x_prime = LeakyReLU(alpha=leakyrelu_alpha)(x_prime)
        # conv 4x4
        x_prime = Conv2D(128, (4, 4), padding='same', kernel_initializer='he_normal')(x_prime)
        x_prime = tf.keras.layers.BatchNormalization()(x_prime)
        x_prime = LeakyReLU(alpha=leakyrelu_alpha)(x_prime)
        # dense output layer
        x_prime = Flatten()(x_prime)
        x_prime = Dense(1)(x_prime)
        self._base_model = Model(x, x_prime)
        # compile base model with adam and mse #
        self._base_model.compile(loss='mse', optimizer=Adam(lr=init_lr, beta_1=init_beta_1, beta_2=init_beta_2, epsilon=init_epsilon))
        # target model
        self._current_model = self._base_model
        # fade in model
        self._fadein_model = self._base_model
        # bool: if in fadein transition == True
        self._fadein_state = False

    def grow(self, n_input_layers=3):
        previous_model = self._current_model
        input_shape = list(previous_model.input.shape)[1:]

        # new input shape will be double size:
        # input shape comes in form [None, n, n, c]
        input_shape_prime = input_shape

        # double size of input
        input_shape_prime[0] *= 2
        input_shape_prime[1] *= 2
        ### new layer ###
        input_prime = Input(shape=input_shape_prime)
        x = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(input_prime)
        x = LeakyReLU(alpha=self.leakyrelu_alpha)(x)
        # define new block
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = LeakyReLU(alpha=self.leakyrelu_alpha)(x)
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = LeakyReLU(alpha=self.leakyrelu_alpha)(x)
        x = AveragePooling2D()(x)
        block_new = x

        # skip the input, 1x1 and activation for the old model
        for i in range(n_input_layers, len(previous_model.layers)):
            x = previous_model.layers[i](x)
        # straight pass for once fade in is complete.
        straight_pass = Model(input_prime, x)

        # Fade in model #
        # downsamples by a factor of 2 ( invert upscale of input)
        # i.e 8x8 -> 4x4
        downsample = AveragePooling2D()(input_prime)
        # connect old input processing to downsampled new input
        block_old = previous_model.layers[1](downsample)
        block_old = previous_model.layers[2](block_old)
        # fade in output of old model input layer with new input
        d = WeightedSum()([block_old, block_new])
        # skip over input, 1x1 conv and activation
        for i in range(n_input_layers, len(previous_model.layers)):
            d = previous_model.layers[i](d)
        fadein = Model(input_prime, d)

        # reassign models to continue growth
        self._base_model = previous_model
        self._current_model = straight_pass
        self._fadein_model = fadein

        # going to be in fadein state right after growth call#
        self._fadein_state = True



class Prog_Generator(prog_model):
    def __init__(self,
                 leakyrelu_alpha=0.2,
                 LR_input_size=(4,4, 3),
                 **kwargs
                 ):
        # call the parent constructor
        super(Prog_Generator, self).__init__(**kwargs)
        self.leakyrelu_alpha = leakyrelu_alpha
        ### Construct base model ###
        # TODO: ensure convert input to LR input shape
        x_LR = Input(shape=LR_input_size)
        # linear scale
        x = Dense(128 * 4 * 4, kernel_initializer='he_normal')(x_LR)
        x = Reshape((4, 4, 128))(x)

        # conv 4x4, input block
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = LeakyReLU(alpha=leakyrelu_alpha)(x)

        # conv 3x3
        x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = LeakyReLU(alpha=self.leakyrelu_alpha)(x)
        # conv 1x1, output block
        y = Conv2D(3, (1, 1), padding='same', kernel_initializer='he_normal')(x)
        # define models
        self._base_model = Model(x_LR, y)
        self._current_model = self._base_model
        self._fadein_model = self._base_model

        # going to be in fadein state right after growth call#
        self._fadein_state = False

    def grow(self):
        previous_model = self._current_model
        # get the last layer of the previous model (avoid 1x1 conv)
        end_of_model = previous_model.layers[-2].output

        # upsample by factor of 2
        upsampling = UpSampling2D()(end_of_model)

        # conv 3x3 1
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(upsampling)
        x = tf.keras.layers.BatchNormalization()(x)
        x = LeakyReLU(alpha=self.leakyrelu_alpha)(x)
        # conv 3x3 2
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = LeakyReLU(alpha=self.leakyrelu_alpha)(x)
        # new output
        x = Conv2D(3, (1, 1), padding='same', kernel_initializer='he_normal')(x)
        # define model
        straight_pass = Model(previous_model.input, x)

        ### merged fade in ###
        # get the output layer from old model
        out_old = previous_model.layers[-1]
        # connect the upsampling to the old output layer
        x_prime = out_old(upsampling)
        # define new output image as the weighted sum of the old and new models
        merged = WeightedSum()([x_prime, x])
        fadein = Model(previous_model.input, merged)

        ### update models ###
        self._base_model = previous_model
        self._current_model = straight_pass
        self._fadein_model = fadein
        self._fadein_state = True



class ProGAN():
    def __init__(self,
                 **kwargs
                 ):

        self.Discriminator = Prog_Discriminator(**kwargs)
        self.Generator = Prog_Generator(**kwargs)

    def set_alpha(self, alpha):
        self.Discriminator.set_alpha(alpha)
        self.Generator.set_alpha(alpha)

    def grow(self):
        self.Discriminator.grow()
        self.Generator.grow()

