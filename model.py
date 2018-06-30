import tensorflow as tf

DEFAULT_DIR = '/tmp/cifar10_model'

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck']
NUM_CLASSES = 10
IMAGE_SIZE = 32
COLOURS = 3

def model_fn(features, labels, mode):
    # input layer
    input_layer = tf.reshape(features['x'], [-1, IMAGE_SIZE, IMAGE_SIZE, COLOURS])

    conv = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)


    flat = tf.reshape(conv, [-1, IMAGE_SIZE * IMAGE_SIZE * 32])

    dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

    logits = tf.layers.dense(inputs=dense, units=10)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    # for prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # calculates loss
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=NUM_CLASSES)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # for training
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # evaluate return
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels,
            predictions=predictions['classes'])
        }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
