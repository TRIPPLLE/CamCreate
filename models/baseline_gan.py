import tensorflow as tf
from tensorflow.keras import layers

def EncoderDecoderGenerator(in_channels=3, out_channels=3, features=64):
    inputs = layers.Input(shape=(None, None, in_channels))
    
    # Encoder
    e1 = layers.Conv2D(features, 4, strides=2, padding='same')(inputs)
    e1 = layers.LeakyReLU(0.2)(e1) # 128
    
    e2 = layers.Conv2D(features*2, 4, strides=2, padding='same', use_bias=False)(e1)
    e2 = layers.BatchNormalization()(e2)
    e2 = layers.LeakyReLU(0.2)(e2) # 64
    
    e3 = layers.Conv2D(features*4, 4, strides=2, padding='same', use_bias=False)(e2)
    e3 = layers.BatchNormalization()(e3)
    e3 = layers.LeakyReLU(0.2)(e3) # 32
    
    e4 = layers.Conv2D(features*8, 4, strides=2, padding='same', use_bias=False)(e3)
    e4 = layers.BatchNormalization()(e4)
    e4 = layers.LeakyReLU(0.2)(e4) # 16
    
    e5 = layers.Conv2D(features*8, 4, strides=2, padding='same', use_bias=False)(e4)
    e5 = layers.BatchNormalization()(e5)
    e5 = layers.LeakyReLU(0.2)(e5) # 8
    
    e6 = layers.Conv2D(features*8, 4, strides=2, padding='same', use_bias=False)(e5)
    e6 = layers.BatchNormalization()(e6)
    e6 = layers.LeakyReLU(0.2)(e6) # 4
    
    e7 = layers.Conv2D(features*8, 4, strides=2, padding='same', use_bias=False)(e6)
    e7 = layers.BatchNormalization()(e7)
    e7 = layers.LeakyReLU(0.2)(e7) # 2
    
    e8 = layers.Conv2D(features*8, 4, strides=2, padding='same')(e7)
    e8 = layers.ReLU()(e8) # 1
    
    # Decoder
    d1 = layers.Conv2DTranspose(features*8, 4, strides=2, padding='same', use_bias=False)(e8)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.ReLU()(d1) # 2
    
    d2 = layers.Conv2DTranspose(features*8, 4, strides=2, padding='same', use_bias=False)(d1)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.ReLU()(d2) # 4
    
    d3 = layers.Conv2DTranspose(features*8, 4, strides=2, padding='same', use_bias=False)(d2)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.ReLU()(d3) # 8
    
    d4 = layers.Conv2DTranspose(features*8, 4, strides=2, padding='same', use_bias=False)(d3)
    d4 = layers.BatchNormalization()(d4)
    d4 = layers.ReLU()(d4) # 16
    
    d5 = layers.Conv2DTranspose(features*4, 4, strides=2, padding='same', use_bias=False)(d4)
    d5 = layers.BatchNormalization()(d5)
    d5 = layers.ReLU()(d5) # 32
    
    d6 = layers.Conv2DTranspose(features*2, 4, strides=2, padding='same', use_bias=False)(d5)
    d6 = layers.BatchNormalization()(d6)
    d6 = layers.ReLU()(d6) # 64
    
    d7 = layers.Conv2DTranspose(features, 4, strides=2, padding='same', use_bias=False)(d6)
    d7 = layers.BatchNormalization()(d7)
    d7 = layers.ReLU()(d7) # 128
    
    d8 = layers.Conv2DTranspose(out_channels, 4, strides=2, padding='same', activation='tanh')(d7) # 256
    
    return tf.keras.Model(inputs=inputs, outputs=d8, name='EncoderDecoderGenerator')

def PatchGANDiscriminator(in_channels=6):
    inputs = layers.Input(shape=(None, None, in_channels))
    
    # 256 -> 128
    x = layers.Conv2D(64, 4, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    
    # 128 -> 64
    x = layers.Conv2D(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # 64 -> 32
    x = layers.Conv2D(256, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # 32 -> 31 (stride 1)
    # TF padding 'same' with stride 1 keeps output size 32. 
    # To mimic PyTorch padding=1 with kernel=4, it shrinks size by -1.
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
    x = layers.Conv2D(512, 4, strides=1, padding='valid', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # 31 -> 30 (stride 1)
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
    out = layers.Conv2D(1, 4, strides=1, padding='valid')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=out, name='PatchGANDiscriminator')

def gan_loss_baseline(D_out, is_real):
    if is_real:
        target = tf.ones_like(D_out)
    else:
        target = tf.zeros_like(D_out)
    return tf.reduce_mean(tf.square(D_out - target))
