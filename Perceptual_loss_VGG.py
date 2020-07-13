"""
Using weights trained on EuroSat,

Implement a VGG for perceptual loss:
referencing: https://arxiv.org/pdf/1609.04802.pdf,
extract all layers with relu activations.
"""
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19



class PROG_PL_VGG19(Model):
    """
    An important note:
    the input for the vgg-19 must be atleast 32x32

    prior to this size, do we truly need perceptual loss?
    """
    def __init__(self, input_dims, layers_to_extract, load_weights, channel_last=True, upscale_factor=2, **kwargs):
        super(PROG_PL_VGG19, self).__init__(**kwargs)

        self.input_dims = input_dims
        self.layers_to_extract = layers_to_extract
        self.load_weights = load_weights
        self.channel_last = channel_last
        self.upscale_factor = upscale_factor
        self._PL_VGG19()

    def _PL_VGG19(self):
        """
        PL -> Perceptual loss
        instantiate pre-trained VGG
        used for feature extraction.
        :return:
        """
        vgg = VGG19(weights=self.load_weights, include_top=False, input_shape=self.input_dims)
        vgg.trainable = False
        outputs = [vgg.layers[i].output for i in self.layers_to_extract]
        self.model = Model([vgg.input], outputs)
        self.model._name = 'feature_extractor'

    def call(self, input, **kwargs):
        return self.model(input, **kwargs)

    def grow(self):
        """
        must grow when output of gan grows
        vgg-19 will be instantiated with output size of
        GAN, when this doubles the VGG-19 input will also
        have to grow.

        This will re instantiate the vgg-19 with pre-trained wieghts and
        new input size.
        :return: None
        """
        # grow (multiply inputs by upscale factor, default 2)
        if isinstance(self.input_dims, tuple):
            self.input_dims = list(self.input_dims)
        if self.channel_last:
            self.input_dims[0] *= self.upscale_factor
            self.input_dims[1] *= self.upscale_factor
        else:
            self.input_dims[1] *= self.upscale_factor
            self.input_dims[2] *= self.upscale_factor

        # reinstantiate vgg19
        self._PL_VGG19()


