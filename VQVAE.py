from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf
from VectorQuantizer import VectorQuantizer
tfd = tfp.distributions

class Encoder(keras.Model):

    def __init__(self, latent_dim=16, **kwargs):
        encoder_inputs = keras.Input(shape=(28, 28, 1))
        x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")(x)
        super().__init__(encoder_inputs, encoder_outputs, name="encoder", **kwargs)

class Decoder(keras.Model):

    def __init__(self, encoder, **kwargs):
        latent_inputs = keras.Input(encoder.output.shape[1:])
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(latent_inputs)
        x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same")(x)
        super().__init__(latent_inputs, decoder_outputs, name="decoder", **kwargs)

class VQVAE(keras.Model):
    
    def __init__(self, latent_dim=16, num_embeddings=64, **kwargs):
        vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
        encoder = Encoder(latent_dim)
        decoder = Decoder(encoder)
        inputs = keras.Input(shape=(28, 28, 1))
        encoder_outputs = encoder(inputs)
        quantized_latents = vq_layer(encoder_outputs)
        reconstructions = decoder(quantized_latents)
        super().__init__(inputs, reconstructions, name="vqvae", **kwargs)