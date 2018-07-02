import tensorflow as tf
from tensorflow.python.keras.datasets import cifar10
import model
import numpy as np

MODEL_DIR = '/tmp/cifar10_model'
BATCH_SIZE = 100
STEPS_PER_TEST = 5000

tf.logging.set_verbosity(tf.logging.INFO)

def main(argv):
    # get model directory
    if len(argv) < 2:
        model_dir = model.DEFAULT_DIR
    else:
        model_dir = argv[1]

    train, evaluate = cifar10.load_data()

    # modify the image data to have values between 0 and 1
    x_train = np.asarray(train[0] / 255.0, dtype=np.float32)
    x_eval = np.asarray(evaluate[0] / 255.0, dtype=np.float32)

    # modify the label data to have data type int32 and in the correct shape
    # e.g. makes [[7],[4],...,[5]] go to [7, 4, 5]
    y_train = np.asarray(train[1], dtype=np.int32)
    y_train = np.reshape(y_train, [-1])

    y_eval = np.asarray(evaluate[1], dtype=np.int32)
    y_eval = np.reshape(y_eval, [-1])

    classifier = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=model_dir)

    # set up logging to store TBD
    log_tensors = {}
    logging_hook = tf.train.LoggingTensorHook(tensors=log_tensors, every_n_iter=50)

    # input function for training
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': x_train},
        y=y_train,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # input function for evaluating
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': x_eval},
        y=y_eval,
        num_epochs=1,
        shuffle=False)

    while True:
        # train model
        classifier.train(input_fn=train_input_fn, steps=STEPS_PER_TEST, hooks=[logging_hook])

        # evaluate model
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

        # save model
        estimator.export_savedmodel(MODEL_DIR)

if __name__ == '__main__':
    tf.app.run()
