import tensorflow as tf

initial_generator = tf.keras.models.load_model('initial_generator_model.h5')

def generate_images(model, num_examples_to_generate, noise_dim):
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    predictions = model(seed, training=False)
    return predictions

num_examples_to_generate = 10
noise_dim = 100

initial_images = generate_images(initial_generator, num_examples_to_generate, noise_dim)
