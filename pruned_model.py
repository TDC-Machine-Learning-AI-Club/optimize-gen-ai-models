import tensorflow as tf

def generate_images(model, num_examples_to_generate, noise_dim):
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    predictions = model(seed, training=False)
    return predictions

num_examples_to_generate = 10
noise_dim = 100


pruned_generator = tf.keras.models.load_model('pruned_generator_model.h5')

pruned_images = generate_images(pruned_generator, num_examples_to_generate, noise_dim)
