def myvae(x_train,y_train,n_samples):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    import tensorflow as tf
    from keras.layers import Input, Dense, Lambda, Layer
    from keras.models import Model
    from keras import backend as K
    from keras import metrics
    from myutil import Dataset,Smote
#from keras.datasets import mnist
    x_train_generator = []
#    data=np.loadtxt('./MNIST_data/yeast.txt',dtype='float32')
    batch_size = 10
    act = 'sigmoid'
    original_dim = x_train.shape[1]
    total_epoch =  int(x_train.shape[0]/batch_size)
    
    x_train = x_train[0:batch_size*total_epoch,:]
    y_train = y_train[0:batch_size*total_epoch]
    latent_dim = 5
    intermediate_dim = int(original_dim*0.6)
    epochs = 25
    epsilon_std = 1.0
#    n_input = x_train.shape[1]

    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim, activation=act)(x)
    z_mean = Dense(latent_dim, activation=act)(h)
    z_log_var = Dense(latent_dim, activation=act)(h)


    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    
# we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation=act)
    #print (decoder_h.output_shape)
#print (decoder_h.input_shape)
    decoder_mean = Dense(original_dim, activation=act)
#print (decoder_mean.output_shape)
#print (decoder_mean.input_shape)

    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)


# Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, **kwargs):
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean):
            xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean)
            self.add_loss(loss, inputs=inputs)
            # We won't actually use the output.
            return x

    y = CustomVariationalLayer()([x, x_decoded_mean])
    vae = Model(x, y)
    vae.compile(optimizer='Adam', loss=None)
    x_test = x_train
#    y_test = y_train

# train the VAE on MNIST digits
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#    x_train = data[:,0:n_input]
#    y_train = data[:,n_input]
#    x_test = data[:,0:n_input]
#    y_test = data[:,n_input]


#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, x_test))

# build a model to project inputs on the latent space
#    encoder = Model(x, z_mean)
    
# display a 2D plot of the digit classes in the latent space
#    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
#    plt.figure(figsize=(6, 6))
#    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
#    plt.colorbar()
#    plt.show()
    
# build a digit generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    
#    xx = np.array([x_test_encoded[:,0],x_test_encoded[:,1]])
#    xx = xx.transpose()
#    s = Smote(xx,100)
#    ge = s.over_sampling()
    
## display a 2D manifold of the digits
#    n = int(pow(n_samples,1/3))  # figure with 15x15 digits
#digit_size = 28
#figure = np.zeros((digit_size * n, digit_size * n))
## linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
## to produce values of the latent variables z, since the prior of the latent space is Gaussian
#    grid_x = norm.ppf(np.linspace(0.1, 0.9, n))
#    grid_x = np.random.standard_normal(n)
#    grid_x = np.random.normal(np.mean(x_test_encoded[:, 0]),np.std(x_test_encoded[:, 0]),n)
#    grid_y = np.random.normal(np.mean(x_test_encoded[:, 1]),np.std(x_test_encoded[:, 1]),n)
#    grid_y = norm.ppf(np.linspace(0.1, 0.9, n))
#    grid_z = norm.ppf(np.linspace(0.1, 0.9, n))
#    grid_y = np.random.standard_normal(n)
#
#    for i, yi in enumerate(grid_x):
#        for j, xi in enumerate(grid_y):
#            for k,zi in enumerate(grid_z):
#                
##            if xi>=0 and yi >=0:
#                
#                z_sample = np.array([[xi, yi,zi]])
##            print([xi,yi])
#                x_decoded = generator.predict(z_sample)
#                x_train_generator.append(x_decoded[0])
##        digit = x_decoded[0].reshape(digit_size, digit_size)
##        figure[i * digit_size: (i + 1) * digit_size,
##               j * digit_size: (j + 1) * digit_size] = digit
#
#plt.figure(figsize=(10, 10))
#plt.imshow(figure, cmap='Greys_r')
#plt.show()
#    gene = generator.predict(ge)
    for i in range(n_samples):
        z_sample = np.random.randn(1,latent_dim)
        x_decoded = generator.predict(z_sample)
        x_train_generator.append(x_decoded[0])
    return x_train_generator