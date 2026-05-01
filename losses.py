import tensorflow as tf
from keras.applications import VGG19

class PerceptualLoss(tf.keras.Model):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load VGG19 with generic imagenet weights
        vgg = VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        
        # We need the outputs from specific layers.
        # Commonly used layers for perceptual loss in VGG19:
        # block1_conv2, block2_conv2, block3_conv4, block4_conv4, block5_conv4
        layer_names = ['block1_conv2', 'block2_conv2', 
                       'block3_conv4', 'block4_conv4', 'block5_conv4']
        outputs = [vgg.get_layer(name).output for name in layer_names]
        
        self.feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=outputs)
        
    def call(self, y_true, y_pred):
        # the inputs are expected to be in [-1, 1]
        # VGG19 expects inputs in [0, 255] and BGR with ImageNet mean subtracted.
        # tf.keras.applications.vgg19.preprocess_input takes care of this.
        
        y_true = (y_true + 1.0) * 127.5
        y_pred = (y_pred + 1.0) * 127.5
        
        y_true = tf.keras.applications.vgg19.preprocess_input(y_true)
        y_pred = tf.keras.applications.vgg19.preprocess_input(y_pred)
        
        real_features = self.feature_extractor(y_true)
        fake_features = self.feature_extractor(y_pred)
        
        loss = 0.0
        for r_feat, f_feat in zip(real_features, fake_features):
            loss += tf.reduce_mean(tf.abs(r_feat - f_feat))
            
        return loss

class CustomGANLosses:
    def __init__(self):
        self.perceptual = PerceptualLoss()
        
    def feature_matching_loss(self, feat_real, feat_fake):
        """ Combats mode collapse by matching discriminator internal features """
        loss = 0.0
        for r, f in zip(feat_real, feat_fake):
            loss += tf.reduce_mean(tf.abs(f - tf.stop_gradient(r)))
        return loss
        
    def edge_loss(self, img1, img2):
        """ Computes the difference between edges in generated and real sketches """
        # Sobel filters
        filter_x = tf.constant([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=tf.float32)
        filter_y = tf.constant([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=tf.float32)
        
        # Reshape to [filter_height, filter_width, in_channels, out_channels]
        # In TF depthwise conv2d, shape is [FH, FW, IN_CHANNELS, CHANNEL_MULTIPLIER]
        filter_x = tf.reshape(filter_x, [3, 3, 1, 1])
        filter_y = tf.reshape(filter_y, [3, 3, 1, 1])
        
        # We need this for 3 channels (RGB). We can tile it.
        filter_x = tf.tile(filter_x, [1, 1, 3, 1])
        filter_y = tf.tile(filter_y, [1, 1, 3, 1])
        
        # Depthwise conv
        edge1_x = tf.nn.depthwise_conv2d(img1, filter_x, strides=[1,1,1,1], padding='SAME')
        edge1_y = tf.nn.depthwise_conv2d(img1, filter_y, strides=[1,1,1,1], padding='SAME')
        
        edge2_x = tf.nn.depthwise_conv2d(img2, filter_x, strides=[1,1,1,1], padding='SAME')
        edge2_y = tf.nn.depthwise_conv2d(img2, filter_y, strides=[1,1,1,1], padding='SAME')
        
        mag1 = tf.sqrt(tf.square(edge1_x) + tf.square(edge1_y) + 1e-6)
        mag2 = tf.sqrt(tf.square(edge2_x) + tf.square(edge2_y) + 1e-6)
        
        return tf.reduce_mean(tf.abs(mag1 - mag2))
