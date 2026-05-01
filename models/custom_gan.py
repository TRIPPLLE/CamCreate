import tensorflow as tf
from tensorflow.keras import layers

class ResBlock(layers.Layer):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.block = tf.keras.Sequential([
            layers.Conv2D(channels, kernel_size=3, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(channels, kernel_size=3, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization()
        ])
        
    def call(self, x, training=False):
        return tf.nn.relu(x + self.block(x, training=training))

class NoiseInjection(layers.Layer):
    def __init__(self, channels):
        super(NoiseInjection, self).__init__()
        self.weight = self.add_weight(name='noise_weight', shape=(1, 1, 1, channels), initializer='zeros', trainable=True)
        
    def call(self, image):
        noise = tf.random.normal(shape=(tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2], 1), dtype=image.dtype)
        return image + tf.cast(self.weight, image.dtype) * noise

class SpatialChannelAttention(layers.Layer):
    def __init__(self, in_channels):
        super(SpatialChannelAttention, self).__init__()
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.max_pool = layers.GlobalMaxPooling2D(keepdims=True)
        
        self.fc = tf.keras.Sequential([
            layers.Conv2D(in_channels // 8, 1, use_bias=False),
            layers.ReLU(),
            layers.Conv2D(in_channels, 1, use_bias=False)
        ])
        
        self.conv_spatial = layers.Conv2D(1, kernel_size=7, padding='same', use_bias=False)
        
    def call(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_out = tf.nn.sigmoid(avg_out + max_out)
        x = x * channel_out
        
        # Spatial Attention
        avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_in = tf.concat([avg_pool_spatial, max_pool_spatial], axis=-1)
        spatial_out = tf.nn.sigmoid(self.conv_spatial(spatial_in))
        x = x * spatial_out
        
        return x

def CustomGenerator(in_channels=3, out_channels=3, features=64):
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
    
    # Attention
    a = SpatialChannelAttention(features*4)(e3)
    
    # ResBlocks
    res = ResBlock(features*4)(a)
    res = ResBlock(features*4)(res)
    res = ResBlock(features*4)(res)
    
    # Multi-scale Decoder
    d1 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(res)
    d1 = layers.Conv2D(features*2, 3, strides=1, padding='same', use_bias=False)(d1)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.ReLU()(d1)
    d1 = NoiseInjection(features*2)(d1)
    out64 = layers.Conv2D(out_channels, 3, strides=1, padding='same', activation='tanh', name='out64')(d1)
    
    d2_in = layers.Concatenate()([d1, e2])
    d2 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d2_in)
    d2 = layers.Conv2D(features, 3, strides=1, padding='same', use_bias=False)(d2)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.ReLU()(d2)
    d2 = NoiseInjection(features)(d2)
    out128 = layers.Conv2D(out_channels, 3, strides=1, padding='same', activation='tanh', name='out128')(d2)
    
    d3_in = layers.Concatenate()([d2, e1])
    d3 = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d3_in)
    d3 = layers.Conv2D(features//2, 3, strides=1, padding='same', use_bias=False)(d3)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.ReLU()(d3)
    d3 = NoiseInjection(features//2)(d3)
    out256 = layers.Conv2D(out_channels, 3, strides=1, padding='same', activation='tanh', name='out256')(d3)
    
    return tf.keras.Model(inputs=inputs, outputs=[out64, out128, out256], name='CustomGenerator')

# For Spectral Normalization in TF, we can use a basic implementation if addons are unavailable
class SpectralNormalization(tf.keras.layers.Wrapper):
    """
    Simple implementation of Spectral Normalization for Keras Dense and Conv2D layers.
    Ref: https://arxiv.org/abs/1802.05957
    """
    def __init__(self, layer, power_iterations=1, **kwargs):
        super(SpectralNormalization, self).__init__(layer, **kwargs)
        self.power_iterations = power_iterations
        
    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            
            if hasattr(self.layer, 'kernel'):
                self.w = self.layer.kernel
            elif hasattr(self.layer, 'embeddings'):
                self.w = self.layer.embeddings
            else:
                raise ValueError('SpectralNorm requires a kernel mapping.')

            w_shape = self.w.shape.as_list()
            self.u = self.add_weight(
                shape=(1, w_shape[-1]),
                initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                trainable=False,
                name='sn_u',
                dtype=self.w.dtype,
            )

        super(SpectralNormalization, self).build()

    def call(self, inputs, training=None):
        w_shape = self.w.shape.as_list()
        w_reshaped = tf.reshape(self.w, [-1, w_shape[-1]])
        
        u = self.u
        for _ in range(self.power_iterations):
            v = tf.math.l2_normalize(tf.matmul(u, w_reshaped, transpose_b=True))
            u = tf.math.l2_normalize(tf.matmul(v, w_reshaped))
            
        sigma = tf.matmul(tf.matmul(v, w_reshaped), u, transpose_b=True)
        self.u.assign(u)
        
        self.layer.kernel.assign(self.w / sigma)
        return self.layer(inputs)

def make_discriminator_net(max_features, n_layers, name):
    inputs = layers.Input(shape=(None, None, 6))
    features = 64
    
    x = SpectralNormalization(layers.Conv2D(features, 4, strides=2, padding='same'))(inputs)
    x = layers.LeakyReLU(0.2)(x)
    
    layer_outputs = [x]
    
    for _ in range(1, n_layers):
        next_features = min(features * 2, max_features)
        x = SpectralNormalization(layers.Conv2D(next_features, 4, strides=2, padding='same'))(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        layer_outputs.append(x)
        features = next_features
        
    out = SpectralNormalization(layers.Conv2D(1, 4, strides=1, padding='same'))(x)
    
    return tf.keras.Model(inputs=inputs, outputs=[out] + layer_outputs, name=name)

class CustomDiscriminator(tf.keras.Model):
    def __init__(self, in_channels=6):
        super(CustomDiscriminator, self).__init__()
        self.global_D = make_discriminator_net(256, 4, name='GlobalDiscriminator')
        self.local_D = make_discriminator_net(128, 2, name='LocalDiscriminator')
        
    def call(self, sketch, photo, training=False):
        x = tf.concat([sketch, photo], axis=-1)
        
        global_out = self.global_D(x, training=training)
        local_out = self.local_D(x, training=training)
        
        out_global = global_out[0]
        out_local = local_out[0]
        
        # Features for feature matching
        ret_features = global_out[1:] + local_out[1:]
        
        return out_global, out_local, ret_features
