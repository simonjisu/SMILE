import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model

import numpy as np

    
class ConvLayer(Layer):
    def __init__(self, filters, kernel_size, padding:str='same'):
        super(ConvLayer, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.conv = layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding)
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.pool = layers.MaxPool2D((2,2))
        
    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

    
class Prototypical_Network(Model):
    def __init__(self, w:int=28, h:int=28, c:int=1):
        super(Prototypical_Network, self).__init__()
        self.w, self.h, self.c = w, h, c

        self.encoder = tf.keras.Sequential([
            ConvLayer(64, 3, 'same'),
            ConvLayer(64, 3, 'same'),
            ConvLayer(64, 3, 'same'),
            ConvLayer(64, 3, 'same'),
            layers.Flatten()
        ])
        
        
    def call(self, support, query):
        n_way = support.shape[0]
        n_support = support.shape[1]
        n_query = query.shape[1]
        
        reshaped_s = tf.reshape(support, (n_way * n_support, self.w, self.h, self.c))
        reshaped_q = tf.reshape(query, (n_way * n_query, self.w, self.h, self.c))
        
        # Embeddings are in the shape of (n_support+n_query, 64)
        embeddings = self.encoder(tf.concat([reshaped_s, reshaped_q], axis=0))
        
        # Support prototypes are in the shape of (n_way, n_support, 64)
        s_prototypes = tf.reshape(embeddings[:n_way * n_support], [n_way, n_support, embeddings.shape[-1]])
        # Find the average of prototypes for each class in n_way
        s_prototypes = tf.math.reduce_mean(s_prototypes, axis=1)
        # Query embeddings are the remainding embeddings
        q_embeddings = embeddings[n_way * n_support:]
        
        
        loss = 0.0
        acc = 0.0
        ############### Your code here ###################
            # TODO: finish implementing this method.
            # For a given task, calculate the Euclidean distance
            # for each query embedding and support prototypes.
            # Then, use these distances to calculate
            # both the loss and the accuracy of the model.
            # HINT: you can use tf.nn.log_softmax()
        
        # calculate euclidean_distance
        # q: (n_way*n_query, n_way[same embeddings], n_emb)
        # s: (n_way*n_query, n_way[different prototypes], n_emb)
        # for each embedding in query, it calculate the euclidean distance
        # to each label (prototypes)
        # euc_dist: (n_way*n_query, n_way)
        q = tf.tile(tf.expand_dims(q_embeddings, 1), [1, n_way, 1])
        s = tf.tile(tf.expand_dims(s_prototypes, 0), [q_embeddings.shape[0], 1, 1])
        euc_dist = tf.reduce_mean(tf.math.pow(q - s, 2), 2)
        # prob: (n_way, n_query, n_way)
        prob = tf.reshape(tf.nn.log_softmax(-euc_dist), [n_way, n_query, -1])
        y = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query)).astype(np.uint8)
        y_one_hot = tf.one_hot(y, n_way)
        loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, prob), axis=-1), [-1]))
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prob, axis=-1), y), tf.float32))

        ##################################################
        
        return loss, acc
    
    
    