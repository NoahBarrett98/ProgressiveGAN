"""
Progressive Growing Gan
The Progressive Growing GAN is an extension to the GAN training procedure that
involves training a GAN to generate very small images, such as 4×4,
and incrementally increasing the size of the generated images to 8×8, 16×16,
until the desired output size is met. This has allowed the progressive GAN to generate
photorealistic synthetic faces with 1024×1024 pixel resolution.

described in the 2017 paper by Tero Karras, et al. from Nvidia
titled “Progressive Growing of GANs for Improved Quality, Stability, and Variation.”

to consider:
    full images are used entire time i.e. an entire scene is reduced to 4x4 image, might make more sense to
    use fragments of the image to get more meaningful representations
"""

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Conv2D, UpSampling2D, AveragePooling2D, LeakyReLU
from Layers import *


class Prog_Discriminator(Model):
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

        # intialize with 512
        self.num_filters = 512

        # to enable reduction of filter size
        self.growth_phase = 0

        ### Construct base model ###
        # Input for first growth
        self.input_conv = Conv2D(self.num_filters, (1, 1), padding='same', kernel_initializer='he_normal')
        self.input_act = LeakyReLU(alpha=0.2)
        self.input_dnsmpl = AveragePooling2D()

        # top to be filled with dis blocks
        self.dis_blocks = []

        # minibatch std dev layer
        self.MinibatchStdev = MinibatchStdev()

        # conv 3x3
        self.conv1 = Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal')
        self.act1 = LeakyReLU(alpha=leakyrelu_alpha)


        # conv 4x4 (1x1 dims)
        self.conv2 = Conv2D(512, (4, 4), padding='same', strides=(512, 512), kernel_initializer='he_normal')
        self.act2 = LeakyReLU(alpha=leakyrelu_alpha)


        # dense output layer
        self.flatten = Flatten()
        self.dense = Dense(1)

        # weighted output
        self.weighted_sum = WeightedSum()

        # intialize with growth period
        self.grow()

    def set_ws_alpha(self, alpha):
        self.weighted_sum.set_alpha(alpha)

    def grow(self):
        num_filters = self.num_filters
        # reduction in filters occurs after 3rd phase
        decrease_filters = self.growth_phase > 2
        # insert new dis block to front of list
        self.dis_blocks.insert(0,
            dis_block(num_filters,
                      decrease_filters)
        )

        self.growth_phase += 1
        if decrease_filters:
            self.num_filters = int(self.num_filters / 2)

        # disable following blocks input section
        if self.growth_phase > 1:
            self.dis_blocks[1].is_top = False
            # disable training for input layers in block.
            self.dis_blocks[1].deactivate_input()

    def call(self, inputs):
        # input block (this may be an erraneous implementation #
        # inputs will potentially need to grow with the model #
        x = inputs

        if self.growth_phase > 1:
            # straight pass
            x_prime = self.dis_blocks[0](x)
            x_prime = self.dis_blocks[1](x_prime)
            # pass through old block as if it is input
            self.dis_blocks[1].is_top = True
            x = self.dis_blocks[1](x)
            self.dis_blocks[1].is_top = False

            # pass through all layers except last layer
            for i, block in enumerate(self.dis_blocks[2:]):
                x = block(x)
                x_prime = block(x_prime)

        else:
            # straight pass, nothing to pass through x
            x_prime = self.dis_blocks[0](x)
            # input activation for first growth phase ( old model )
            x = self.input_conv(x)
            x = self.input_act(x)

        x = self.input_dnsmpl(x)

        # straight pass cont'd
        x_prime = self.MinibatchStdev(x_prime)
        x_prime = self.conv1(x_prime)
        x_prime = self.act1(x_prime)
        x_prime = self.conv2(x_prime)

        x_prime = self.act2(x_prime)
        x_prime = self.flatten(x_prime)
        x_prime = self.dense(x_prime)

        # old pass
        x = self.MinibatchStdev(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.flatten(x)
        x = self.dense(x)

        # fade in two outputs
        x = self.weighted_sum([x, x_prime])

        return x


class Prog_Generator(Model):
    def __init__(self,
                 leakyrelu_alpha=0.2,
                 LR_input_size=(4, 4, 3),
                 kernel_initializer='he_normal',
                 **kwargs
                 ):
        # call the parent constructor
        super(Prog_Generator, self).__init__(**kwargs)

        self.leakyrelu_alpha = leakyrelu_alpha
        self.LR_input_size = LR_input_size
        self.kernel_initializer = kernel_initializer

        # intialize with 512
        self.num_filters = 512

        # to enable reduction of filter size
        self.growth_phase = 0

        ### Construct base model ###
        # input = Input(shape=self.LR_input_size)
        # conv 4x4, input block
        self.conv1 = Conv2D(512, (4, 4), padding='same', kernel_initializer=self.kernel_initializer)
        self.act1 = LeakyReLU(alpha=self.leakyrelu_alpha)

        # conv 3x3, input block
        self.conv2 = Conv2D(512, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)
        self.act2 = LeakyReLU(alpha=self.leakyrelu_alpha)

        # center to be filled with gen blocks
        self.gen_blocks = []

        # output block #
        # upsample
        self.upspl_last = UpSampling2D()
        # conv 3x3

        self.conv_last1 = Conv2D(16, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)
        self.act_last1 = LeakyReLU(alpha=self.leakyrelu_alpha)
        # conv 3x3
        self.conv_last2 = Conv2D(16, (3, 3), padding='same', kernel_initializer=self.kernel_initializer)
        self.act_last2 = LeakyReLU(alpha=self.leakyrelu_alpha)
        # weighted sum for merging of outputs
        self.weighted_sum = WeightedSum()
        # conv 1x1, output block
        self.RGB_out =  Conv2D(3, (1, 1), padding='same', kernel_initializer=self.kernel_initializer)

        # intialize with growth period
        self.grow()

    def set_ws_alpha(self, alpha):
        self.weighted_sum.set_alpha(alpha)

    def grow(self):

        num_filters = self.num_filters
        # reduction in filters occurs after 3rd phase
        reduce_filters = self.growth_phase < 2
        # add new gen block to model
        self.gen_blocks.append(
                                gen_block(num_filters,
                                          reduce_filters)
        )
        # remove upsamples as growth occurs
        # this will help compensate for input growth (img size increase)
        if self.growth_phase >= 1:
            self.gen_blocks[self.growth_phase-1].upsample = False

        self.growth_phase += 1
        if reduce_filters:
            self.num_filters= int(self.num_filters/2)

        if self.growth_phase > 1:
            # new block so remove end block from prev block
            self.gen_blocks[-2].is_end = False
            # ensure we are not training params we should not be
            self.gen_blocks[-2].deactivate_output()

    def call(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)

        # pass through all layers except last layer
        for block in self.gen_blocks[:-1]:
            x = block(x)

        # intializes with growth period so always in growth phase
        # straight pass, this will be RGBout
        x_prime = self.gen_blocks[-1](x)


        # old pass
        x = self.upspl_last(x)
        x = self.conv_last1(x)
        x = self.conv_last2(x)
        x = self.act_last2(x)
        x = self.RGB_out(x)

        # fade in two outputs
        x = self.weighted_sum([x, x_prime])

        return x

class ProGAN(object):
    def __init__(self,
                 **kwargs
                 ):

        self.Discriminator = Prog_Discriminator(**kwargs)
        self.Generator = Prog_Generator(**kwargs)

    def set_alpha(self, alpha):
        self.Discriminator.set_ws_alpha(alpha)
        self.Generator.set_ws_alpha(alpha)

    def grow(self):
        self.Discriminator.grow()
        self.Generator.grow()

