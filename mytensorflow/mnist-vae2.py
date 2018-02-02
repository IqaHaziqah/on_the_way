def mnist_vae(data,gene_size,para_o):
import numpy as np
import keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K
from keras import objectives,optimizers

np.random.seed(1111)
import tensorflow as tf
from myutil2 import Dataset
import numpy as np
from scipy.stats import norm
from myutil2 import random_walk,xavier_init
from keras import objectives
mnist = Dataset(data)
input_dim = data.shape[1]
hidden_encoder_dim = feed_dict['hidden_encoder_dim']
hidden_decoder_dim = feed_dict['hidden_decoder_dim']
latent_dim = feed_dict['latent_dim']
lam = feed_dict['lam']
epochs = feed_dict['epochs']
batch_size = feed_dict['batch_size']
learning_rate = feed_dict['learning_rate']
ran_walk = feed_dict['ran_walk']
check = feed_dict['check']
ini = feed_dict['initial']
ACT = feed_dict['activation']
OPT = feed_dict['optimizer']
normal = feed_dict['norm']
decay = feed_dict['decay']

tf.set_random_seed(42)