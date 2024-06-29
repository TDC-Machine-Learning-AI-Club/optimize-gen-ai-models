"""

STEPS:
1. Set Up the Environment:

Install necessary libraries: TensorFlow or PyTorch, and any other required packages for pruning and quantization.

2. Implement and Train a Simple GAN:
Define the generator and discriminator models.
Train the GAN on the MNIST dataset.

3. Apply Model Pruning:
Prune the trained GAN model to remove redundant weights.

4. Apply Model Quantization:
Quantize the pruned model to reduce precision and size.

5. Assess the Difference:
Compare the performance of the original, pruned, and quantized models in terms of size, speed, and image generation quality.

"""


"""

STEPS:
1. Set Up the Environment:

Install necessary libraries: TensorFlow or PyTorch, and any other required packages for pruning and quantization.

2. Implement and Train a Simple GAN:
Define the generator and discriminator models.
Train the GAN on the MNIST dataset.

3. Apply Model Pruning:
Prune the trained GAN model to remove redundant weights.

4. Apply Model Quantization:
Quantize the pruned model to reduce precision and size.

5. Assess the Difference:
Compare the performance of the original, pruned, and quantized models in terms of size, speed, and image generation quality.

"""


import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Load MNIST dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Create a data generator
def data_generator(images, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(buffer_size=BUFFER_SIZE)
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = data_generator(train_images, BATCH_SIZE)

# Define the generator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

# Define the discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# Loss and optimizers
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training loop
EPOCHS = 1
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

# Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
        print(f'Epoch {epoch+1}/{epochs} completed')

# Training Started
train(train_dataset, EPOCHS)

# Save the initial model
generator.save('initial_generator_model.h5')
discriminator.save('initial_discriminator_model.h5')

# APPLY MODEL PRUNING
import tensorflow_model_optimization as tfmot

# Define the pruning schedule
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                            final_sparsity=0.90,
                                                            begin_step=2000,
                                                            end_step=6000)
}

# Apply pruning to the generator and discriminator models
pruned_generator = tfmot.sparsity.keras.prune_low_magnitude(generator, **pruning_params)
pruned_discriminator = tfmot.sparsity.keras.prune_low_magnitude(discriminator, **pruning_params)

# Compile the pruned models
pruned_generator.compile(optimizer=generator_optimizer, loss=generator_loss)
pruned_discriminator.compile(optimizer=discriminator_optimizer, loss=discriminator_loss)

# Re-train the pruned models
def train_pruned_models(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            noise = tf.random.normal([BATCH_SIZE, noise_dim])
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = pruned_generator(noise, training=True)
                real_output = pruned_discriminator(image_batch, training=True)
                fake_output = pruned_discriminator(generated_images, training=True)
                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, pruned_generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, pruned_discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, pruned_generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, pruned_discriminator.trainable_variables))

        print(f'Epoch {epoch+1}/{epochs} completed')

train_pruned_models(train_dataset, EPOCHS)

# Strip pruning wrappers
pruned_generator = tfmot.sparsity.keras.strip_pruning(pruned_generator)
pruned_discriminator = tfmot.sparsity.keras.strip_pruning(pruned_discriminator)

# Save the pruned models
pruned_generator.save('pruned_generator_model.h5')
pruned_discriminator.save('pruned_discriminator_model.h5')

# APPLY MODEL QUANTIZATION
# Quantize the pruned models
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_generator)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_generator_model = converter.convert()

converter = tf.lite.TFLiteConverter.from_keras_model(pruned_discriminator)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_discriminator_model = converter.convert()

# Save the quantized models
with open('quantized_generator.tflite', 'wb') as f:
    f.write(quantized_tflite_generator_model)

with open('quantized_discriminator.tflite', 'wb') as f:
    f.write(quantized_tflite_discriminator_model)

# Load and Use the Models to Generate Images

# Load the Initial Model
initial_generator = tf.keras.models.load_model('initial_generator_model.h5')

def generate_images(model, num_examples_to_generate, noise_dim):
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    predictions = model(seed, training=False)
    return predictions

initial_images = generate_images(initial_generator, num_examples_to_generate, noise_dim)

# Load the Pruned Model
pruned_generator = tf.keras.models.load_model('pruned_generator_model.h5')

pruned_images = generate_images(pruned_generator, num_examples_to_generate, noise_dim)

# Load the Quantized Model
import tensorflow.lite as tflite

def load_quantized_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

quantized_generator = load_quantized_model('quantized_generator.tflite')

def generate_quantized_images(interpreter, num_examples_to_generate, noise_dim):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    seed = tf.random.normal([num_examples_to_generate, noise_dim]).numpy().astype('float32')
    
    interpreter.set_tensor(input_details[0]['index'], seed)
    interpreter.invoke()
    
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions

quantized_images = generate_quantized_images(quantized_generator, num_examples_to_generate, noise_dim)





