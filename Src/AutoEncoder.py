import keras
from keras import layers
import pandas as pd

data_path = 'normalize_data(minMax)(5-180).csv'  #normalize_data(minMax).csv  #'Data(990712).csv' #Data(990712)(without Powdr specification ).csv
dataframe = pd.read_csv(data_path, encoding='ISO-8859–1')

# This is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# This is our input image
input_img = keras.Input(dataframe.shape[1])
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(dataframe.shape[1], activation='sigmoid')(encoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)
autoencoder.compile()
autoencoder.fit()
