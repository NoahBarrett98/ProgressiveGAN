from ProGANVanilla import *
from Perceptual_loss_VGG import PL_VGG19
from util import *

import tensorflow as tf
import tensorflow_datasets as tfds
import os

"""
Training Progressive GAN, 
"""
### HYPERPARAMS ###
batch_size = 16
epochs = 256

# image #
UP_SAMPLE = 2 # factor for upsample
START_INPUT_DIM = 2 # start with 2x2 input -> initialize with growth phase to 4x4 (so really 4)
TARGET_DIM = 256 # full image size

# Adam #
lr=0.001
beta_1=0
beta_2=0.99
epsilon=10e-8


### MODELS ###
ProGAN = ProGAN()

# use 1st conv block for content loss
vgg = PL_VGG19(patch_size=64,
               layers_to_extract=[0,1,2],
               weights='imagenet').model

### DATA ###
"""
NWPU-RESISC45
This dataset requires you to download the source data manually 
into download_config.manual_dir (defaults to ~/tensorflow_datasets/manual/):

Note: this dataset does not have a test/train split.
"""
# load data #
data, info = tfds.load('resisc45', split="train", with_info=True)
# visualize data #
tfds.show_examples(data, info)

# size of entire dataset #
ds_size = info.splits["train"].num_examples
image_shape = info.features['image'].shape
print(image_shape)
# manually split ds into 80:20, train & test respectively #
test_ds_size = int(ds_size*0.20)
train_ds_size = ds_size - test_ds_size
# split #
test_ds = data.take(test_ds_size)
train_ds = data.skip(test_ds_size)
print("size of test: {}, size of train: {}".format(test_ds_size, train_ds_size))

# num features
num_features = info.features["label"].num_classes

# minibatch
test_ds = test_ds.batch(batch_size).repeat(epochs)
train_ds = train_ds.batch(batch_size).repeat(epochs)

# convert to np array
test_ds = tfds.as_numpy(test_ds)
train_ds = tfds.as_numpy(train_ds)


"""
A training batch will consist of generation of image for each sample,
train discrim on both generated images and real ones. 

1:2 ratio of samples 
"""

# update the alpha value on each instance of WeightedSum
def update_fadein(step):
    # calculate current alpha (linear from 0 to 1)
    # we only perform fadein in training #
    alpha = step / float(train_ds_size - 1)
    # update the alpha for each model
    ProGAN.set_alpha(alpha)
    
### loss functions ###
gen_loss = tf.keras.losses.BinaryCrossentropy()
discrim_loss = tf.keras.losses.BinaryCrossentropy()

### adam optimizer for SGD ###
optimizer = tf.keras.optimizers.Adam(lr=lr,
                                    beta_1=beta_1,
                                    beta_2=beta_2,
                                    epsilon=epsilon)

### intialize train metrics ###
gen_train_loss = tf.keras.metrics.Mean(name='gen_train_loss')
gen_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='gen_train_accuracy')
dis_train_loss = tf.keras.metrics.Mean(name='dis_train_loss')
dis_train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='dis_train_accuracy')

### intialize test metrics ###
gen_test_loss = tf.keras.metrics.Mean(name='gen_test_loss')
gen_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='gen_test_accuracy')
dis_test_loss = tf.keras.metrics.Mean(name='dis_test_loss')
dis_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='dis_test_accuracy')

### generator train step ###
@tf.function
def gen_train_step(high_res, low_res, fadein):
  with tf.GradientTape() as tape:
    # training=True is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = ProGAN.Generator(low_res, training=True, fadein=fadein)
    # mean squared error in prediction
    m_loss = tf.keras.losses.MSE(high_res, predictions)

    # content loss
    v_pass = vgg(high_res)
    v_loss = tf.keras.losses.MSE(v_pass, predictions)

    # GAN loss + mse loss + feature loss
    loss = gen_loss(high_res, predictions) + v_loss + m_loss

  # update either current or fadein
  if fadein:
      model = ProGAN.Generator._fadein_model
  else:
      model = ProGAN.Generator._current_model

  # apply gradients
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  # update metrics
  gen_train_loss(loss)
  gen_train_accuracy(model, predictions)


### discriminator train step ###
@tf.function
def dis_train_step(high_res, low_res, step, fadein):
    with tf.GradientTape() as tape:
        # discrim is a simple conv that perfroms binary classification
        # either SR or HR
        # use super res on even, true image on odd steps #
        if step % 2:
            x = ProGAN.Generator(low_res, training=False)
        else:
            x = high_res
        # predict on gen output
        predictions = ProGAN.Discriminator(x, training=True, fadein=fadein)
        loss = discrim_loss(high_res, predictions)

    # update either current or fadein
    if fadein:
        model = ProGAN.Discriminator._fadein_model
    else:
        model = ProGAN.Discriminator._current_model

    # apply gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # update metrics
    gen_train_loss(loss)
    gen_train_accuracy(model, predictions)


### generator test step ###
@tf.function
def gen_test_step(high_res, low_res, fadein):
  # feed test sample in
  predictions = ProGAN.Generator(low_res, training=False, fadein=fadein)
  t_loss = gen_loss(high_res, predictions)

  # update metrics
  gen_test_loss(t_loss)
  gen_test_accuracy(high_res, predictions)


### discriminator test step ###
@tf.function
def dis_test_step(high_res, low_res, step, fadein):
    # feed test sample in
    # use super res on even, true image on odd steps #
    if step % 2:
        x = ProGAN.Generator(low_res, training=False)
    else:
        x = high_res
    # predict on gen output
    predictions = ProGAN.Discriminator(x, training=False, fadein=fadein)
    t_loss = gen_loss(high_res, predictions)

    # update metrics
    gen_test_loss(t_loss)
    gen_test_accuracy(high_res, predictions)



### TRAIN ###
def train(epoch, fadein):
    """
    train step
    :param epoch: int epoch
    :param fadein: bool True for fadein (training)
    :return:
    """
    # Reset the metrics at the start of the next epoch
    gen_train_loss.reset_states()
    gen_train_accuracy.reset_states()
    gen_test_loss.reset_states()
    gen_test_accuracy.reset_states()

    dis_train_loss.reset_states()
    dis_train_accuracy.reset_states()
    dis_test_loss.reset_states()
    dis_test_accuracy.reset_states()

    # alternating training pattern
    if epoch % 2:
        # train generator on even epochs
        # iterator split into batches
        # apply alpha in training #
        for j, batch in enumerate(train_ds):
            for i, sample in enumerate(batch):
                if fadein:
                    # update fade in for given step #
                    update_fadein(j+i)
                high_res, low_res = preprocess(sample, input_dim, UP_SAMPLE)
                gen_train_step(high_res, low_res, fadein)

        # iterator split into batches
        for batch in test_ds:
            for sample in batch:
                test_high_res, test_low_res = preprocess(sample, input_dim, UP_SAMPLE)
                gen_test_step(test_high_res, test_low_res, fadein)

        template = 'Training Generator:\nEpoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              gen_train_loss.result(),
                              gen_train_accuracy.result() * 100,
                              gen_test_loss.result(),
                              gen_test_accuracy.result() * 100))
    else:
        # train discriminator on odd epochs
        for j, batch in enumerate(train_ds):
            for i, sample in enumerate(batch):
                if fadein:
                    # update fade in for given step #
                    update_fadein(j+i)

                high_res, low_res = preprocess(sample, input_dim, UP_SAMPLE)
                dis_train_step(high_res, low_res, i, fadein)

        for batch in test_ds:
            for i, sample in enumerate(batch):
                test_high_res, test_low_res = preprocess(sample, input_dim, UP_SAMPLE)
                dis_test_step(test_high_res, test_low_res, i, fadein)

        template = 'Training Discriminator:\nEpoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              dis_train_loss.result(),
                              dis_train_accuracy.result() * 100,
                              dis_test_loss.result(),
                              dis_test_accuracy.result() * 100))



# initialize input_dim
input_dim = START_INPUT_DIM

# intialize growth phase (input = 2 -> input = 4)
ProGAN.grow()
input_dim*=UP_SAMPLE

while input_dim <= TARGET_DIM:
    # train on given input dim #
    for epoch in range(epochs):

        # train fadein #
        train(epoch, fadein=True)

        # train straight through #
        train(epoch, fadein=False)

    # grow input #
    ProGAN.grow() # upsamples by factor of 2
    input_dim*=UP_SAMPLE # increase input by upsample factor (2)

"""
@article{Cheng_2017,
   title={Remote Sensing Image Scene Classification: Benchmark and State of the Art},
   volume={105},
   ISSN={1558-2256},
   url={http://dx.doi.org/10.1109/JPROC.2017.2675998},
   DOI={10.1109/jproc.2017.2675998},
   number={10},
   journal={Proceedings of the IEEE},
   publisher={Institute of Electrical and Electronics Engineers (IEEE)},
   author={Cheng, Gong and Han, Junwei and Lu, Xiaoqiang},
   year={2017},
   month={Oct},
   pages={1865-1883}
}
"""