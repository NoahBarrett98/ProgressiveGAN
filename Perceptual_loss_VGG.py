"""
Using weights trained on EuroSat,

Implement a VGG for perceptual loss:
referencing: https://arxiv.org/pdf/1609.04802.pdf,
extract all layers with relu activations.
"""
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import VGG19



class PL_VGG19:

    def __init__(self, patch_size, layers_to_extract, weights):
        self.patch_size = patch_size
        self.input_shape = (patch_size,) * 2 + (3,)
        self.layers_to_extract = layers_to_extract
        self.weights = weights
        self._PL_VGG19()

    def _PL_VGG19(self):
        """
        instantiate pre-trained VGG
        used for feature extraction.
        :return:
        """
        vgg = VGG19(weights=self.weights, include_top=False, input_shape=self.input_shape)
        vgg.trainable = False
        outputs = [vgg.layers[i].output for i in self.layers_to_extract]
        self.model = Model([vgg.input], outputs)
        self.model._name = 'feature_extractor'
        self.name = 'vgg19'  # used in weights naming

    #TODO: make growth function vgg input
