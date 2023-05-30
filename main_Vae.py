import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from VAE import VAE
from conVAE import conVAE

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
data_variance = np.var(mnist_digits)

def show_anim(vae: VAE, labels):
    z_mean_values = np.array(vae.z_mean_values)
    fig, ax = plt.subplots()
    scatter = ax.scatter([], [])
    ax.set_xlim((-3.5, 3.5))
    ax.set_ylim((-3.5, 3.5))

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        return (scatter,)

    def update(frame):
        scatter.set_offsets(z_mean_values[frame])
        scatter.set_array(labels)
        return (scatter,)

    animation = FuncAnimation(fig, update, frames=len(z_mean_values), init_func=init, blit=True, interval=50)
    plt.show()
    animation.save('./results/plot.mkv', writer="ffmpeg")
    animation.save('./results/plot.gif', writer="pillow")

if input("Train? y/n ") == "y":
    if input("For conVAE => c, For basic VAE => v: ") == "c":
        vae = conVAE(data_variance, 0.7)
    else:
        vae = VAE(data_variance, 0.4)
    vae.compile(optimizer=keras.optimizers.Adam(), run_eagerly=True)
    vae.fit(mnist_digits, epochs=10, batch_size=64)
    vae.decoder.save('./models/vae_decoder_L2_2')
    vae.encoder.save('./models/vae_encoder_L2_2')
    decoder = vae.decoder
    encoder = vae.encoder
    show_anim(vae, y_train)
else:
    decoder = keras.models.load_model('./models/vae_decoder_bincross_2')
    encoder = keras.models.load_model('./models/vae_encoder_bincross_2')


def plot_latent_space(decoder, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 28
    scale = 3.5
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()

plot_latent_space(decoder)

def plot_label_clusters(encoder, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

x_train = np.expand_dims(x_train, -1).astype("float32") / 255

plot_label_clusters(encoder, x_train, y_train)

def get_vae_likelihood(encoder, decoder, data):
    z_mean, z_log_var, z = encoder.predict(data)
    reconstruction = decoder.predict(z)
    reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    total_loss = reconstruction_loss + kl_loss

    # Calculate posterior probability p(x|z)
    print(tf.reduce_mean(tf.exp(-total_loss)))

# get_vae_likelihood(encoder, decoder, x_train)

def show_reconstruct(encoder, decoder, data):
    z_mean, z_log_var, z = encoder.predict(data)
    reconstruction = decoder.predict(z)

    for i in range(4):
        plt.subplot(1, 2, 1)
        plt.imshow(data[i].squeeze() + 0.5)
        plt.title("Random code")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(reconstruction[i])
        plt.title("Generated image")
        plt.axis("off")
        plt.show()