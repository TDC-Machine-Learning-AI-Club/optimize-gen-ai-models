import tensorflow.lite as tflite
import tensorflow as tf

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


def generate_images(model, num_examples_to_generate, noise_dim):
    seed = tf.random.normal([num_examples_to_generate, noise_dim])
    predictions = model(seed, training=False)
    return predictions

num_examples_to_generate = 10
noise_dim = 100

quantized_images = generate_quantized_images(quantized_generator, num_examples_to_generate, noise_dim)
