# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 19:43:26 2018

@author: zhouying
"""
def mnist_vae(data,gene_size,feed_dict):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm    
    from keras.layers import Input, Dense, Lambda, Layer
    from keras.models import Model
    from keras import backend as K
    from keras import metrics
    from keras import optimizers
    
        
    
    original_dim = data.shape[1]
    batch_size = feed_dict['batch_size']
    latent_dim = feed_dict['latent_dim']
    intermediate_dim = feed_dict['hidden_encoder_dim']
    epochs = feed_dict['epochs']
    epsilon_std = 1.0
    
    
    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim,kernel_initializer='he_uniform', activation='relu')(x)
    z_mean = Dense(latent_dim,kernel_initializer='he_uniform')(h)
    z_log_var = Dense(latent_dim,kernel_initializer='he_uniform')(h)
    
    
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon
    
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim,kernel_initializer='he_uniform',activation='relu')
    decoder_mean = Dense(original_dim,kernel_initializer='he_uniform')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    
    
    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)
    
        def vae_loss(self, x, x_decoded_mean):
            xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
            mes_loss = original_dim*metrics.mean_squared_error(x,x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(mes_loss + kl_loss)
    
        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            # We won't actually use the output.
            return x
    
    y = CustomVariationalLayer()([x, x_decoded_mean])
    vae = Model(x, y)
    opt = optimizers.Adam(lr=0.001)
    vae.compile(optimizer='Adam', loss=None)
    
    
    # train the VAE on MNIST digits
    x_train = data
    x_test = data
    
    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            verbose=2,
            validation_data=(x_test, None))
    
    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)
    
    # display a 2D plot of the digit classes in the latent space
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
    plt.show()
    
    # build a digit generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    
    # display a 2D manifold of the digits
      # figure with 15x15 digits
    # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
    # to produce values of the latent variables z, since the prior of the latent space is Gaussian
    gene = []
    for value in gene_size:
        n = int(np.sqrt(value))
        grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
        x_decoded = []
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):
                z_sample = np.array([[xi, yi]])
                x_decoded.append(generator.predict(z_sample))
        x_decoded = np.array(x_decoded)
        gene.append(x_decoded.squeeze(axis=1))
    return encoder.get_weights(),generator.get_weights()
    
