import numpy as np
import matplotlib.pyplot as plt
from VectorQuantizer import get_code_indices
from VQVAETrainer import VQVAETrainer
from tensorflow import keras
import tensorflow as tf

def show_subplot(original, reconstructed, code=None, real_code=None):
    plt.subplot(1, 4, 1)
    plt.imshow(original.squeeze() + 0.5)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(reconstructed.squeeze() + 0.5)
    plt.title("Reconstructed")
    plt.axis("off")

    if real_code != None:
        plt.subplot(1, 4, 2)
        plt.imshow(real_code.numpy().reshape((7, 7)))
        plt.title("Code before")
        plt.axis("off")

    if code != None:
        plt.subplot(1, 4, 4)
        plt.imshow(code.numpy().reshape((7, 7)))
        plt.title("Code after")
        plt.axis("off")

    plt.show()

def combine(encoded_outputs, decoder, quantizer, dim="width"):
    # Combine two diffent latentn vectors
    dimension = 1 if dim == "width" else 0 
    i = np.random.randint(0, len(encoded_outputs))
    j = np.random.randint(0, len(encoded_outputs))

    encoded_tensor1 = encoded_outputs[i]
    encoded_tensor2 = encoded_outputs[j]

    split_size = encoded_tensor1.shape[1] // 2
    if dim == "width":
        half1_tensor1 = encoded_tensor1[:, :split_size, :]
        half2_tensor1 = encoded_tensor1[:, split_size:, :]
        half1_tensor2 = encoded_tensor2[:, :split_size, :]
        half2_tensor2 = encoded_tensor2[:, split_size:, :]
    elif dim == "height":
        half1_tensor1 = encoded_tensor1[:split_size, :, :]
        half2_tensor1 = encoded_tensor1[split_size:, :, :]
        half1_tensor2 = encoded_tensor2[:split_size, :, :]
        half2_tensor2 = encoded_tensor2[split_size:, :, :]
    combined_tensor1 = np.reshape(tf.concat([half1_tensor1, half2_tensor2], axis=dimension), (1, 7, 7, 16))
    combined_tensor2 = np.reshape(tf.concat([half1_tensor2, half2_tensor1], axis=dimension), (1, 7, 7, 16))

    reconstructed_image1 = decoder.predict(quantizer(combined_tensor1))
    reconstructed_image2 = decoder.predict(quantizer(combined_tensor2))

    plt.subplot(2, 2, 1)
    plt.imshow(test_images[i].squeeze() + 0.5)
    plt.title("First image")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(test_images[j].squeeze() + 0.5)
    plt.title("Second image")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(reconstructed_image1.squeeze() + 0.5)
    plt.title("First combination")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(reconstructed_image2.squeeze() + 0.5)
    plt.title("Second combination")
    plt.axis("off")
    plt.show()

def show_codes(encoded_outputs, quantizer, test_images):
    flat_enc_outputs = encoded_outputs.reshape(-1, encoded_outputs.shape[-1])
    codebook_indices = get_code_indices(quantizer, flat_enc_outputs)
    codebook_indices = codebook_indices.numpy().reshape(encoded_outputs.shape[:-1])

    for i in range(len(test_images)):
        plt.subplot(1, 2, 1)
        plt.imshow(test_images[i].squeeze() + 0.5)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(codebook_indices[i])
        plt.title("Code")
        plt.axis("off")
        plt.show()
    
def show_mupltiple_plots(test_images, reconstructions_test):
    for test_image, reconstructed_image in zip(test_images, reconstructions_test):
        show_subplot(test_image, reconstructed_image)

def shift_encoded_tensor(encoded_outputs, pixels_to_shift, decoder, quantizer, test_images, ax=1):
    # Shift the encoded tensor
    i = np.random.randint(0, len(encoded_outputs))
    shifted_tensor = np.roll(encoded_outputs[i], pixels_to_shift, axis=ax)
    encoded_output = np.reshape(encoded_outputs[i], (1, 7, 7, 16))
    shifted_tensor = np.reshape(shifted_tensor, (1, 7, 7, 16))
    flat_encoded_output = encoded_output.reshape(-1, encoded_output.shape[-1])
    flat_shifted_tensor = shifted_tensor.reshape(-1, shifted_tensor.shape[-1])
    real_code = get_code_indices(quantizer, flat_encoded_output)
    code = get_code_indices(quantizer, flat_shifted_tensor)
    reconstructed_image = decoder.predict(shifted_tensor)

    show_subplot(test_images[i], reconstructed_image, code, real_code)

if __name__=="__main__":
    latent_size = 16
    num_embeddins = 128

    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    x_train_scaled = (x_train / 255.0) - 0.5
    x_test_scaled = (x_test / 255.0) - 0.5

    data_variance = np.var(x_train / 255.0)

    if input("Train? y/n ") == "y":
        vqvae_trainer = VQVAETrainer(data_variance, latent_dim=latent_size, num_embeddings=num_embeddins)
        vqvae_trainer.compile(optimizer=keras.optimizers.Adam())
        vqvae_trainer.fit(x_train_scaled, epochs=10, batch_size=128)
        trained_vqvae_model = vqvae_trainer.vqvae
        trained_vqvae_model.save('./models/vqvaemodel_5')
    else:
        trained_vqvae_model = keras.models.load_model('./models/vqvaemodel')

    idx = np.random.choice(len(x_test_scaled), 10)
    test_images = x_test_scaled[idx]
    reconstructions_test = trained_vqvae_model.predict(test_images)
    trained_vqvae_model.summary()

    encoder = trained_vqvae_model.get_layer("encoder")
    quantizer = trained_vqvae_model.get_layer("vector_quantizer")
    decoder = trained_vqvae_model.get_layer("decoder")

    encoded_outputs = encoder.predict(test_images)

    # show_mupltiple_plots(test_images, reconstructions_test)
    # show_codes(encoded_outputs, quantizer, test_images)
    for _ in range(3):
        combine(encoded_outputs, decoder, quantizer, dim="height")
    for _ in range(5):
        shift_encoded_tensor(encoded_outputs, 10, decoder, quantizer, test_images, 0)