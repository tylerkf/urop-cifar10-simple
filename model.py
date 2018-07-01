import tensorflow as tf

DEFAULT_DIR = '/tmp/cifar10_model'

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck']
NUM_CLASSES = 10
IMAGE_SIZE = 32
COLOURS = 3

LEARNING_RATE = 0.001

def model_fn(features, labels, mode):
    batch_size = features['x'].get_shape().as_list()[0]

    # input layer
    # to [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, COLOURS]
    input_layer = tf.reshape(features['x'], [-1, IMAGE_SIZE, IMAGE_SIZE, COLOURS])

    # 2D convolutional layer - 64 filters, 5x5 kernels
    # to [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 64]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)

    # 2D max pooling layer - 3x3 pooling window, 2x2 strides
    # to [BATCH_SIZE, IMAGE_SIZE / 2, IMAGE_SIZE / 2, 64]
    max_pooling1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[3, 3],
        strides=[2, 2],
        padding='same')

    # local response normalisation
    # to [BATCH_SIZE, IMAGE_SIZE / 2, IMAGE_SIZE / 2, 64]
    lr_normalization1 = tf.nn.local_response_normalization(
        input=max_pooling1,
        depth_radius=4,
        bias=1.0,
        alpha=0.001 / 9.0,
        beta=0.75)

    # 2D convolutional layer - 64 filter, 5x5 kernels
    # to [BATCH_SIZE, IMAGE_SIZE / 2, IMAGE_SIZE / 2, 64]
    conv2 = tf.layers.conv2d(
        inputs=lr_normalization1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)

    # local response normalisation
    # to [BATCH_SIZE, IMAGE_SIZE / 2, IMAGE_SIZE / 2, 64]
    lr_normalization2 = tf.nn.local_response_normalization(
        input=conv2,
        depth_radius=4,
        bias=1.0,
        alpha=0.001 / 9.0,
        beta=0.75)

    # 2D max pooling layer - 3x3 pooling window, 2x2 strides
    # to [BATCH_SIZE, IMAGE_SIZE / 4, IMAGE_SIZE / 4, 64]
    max_pooling2 = tf.layers.max_pooling2d(
        inputs=lr_normalization2,
        pool_size=[3, 3],
        strides=[2, 2],
        padding='same')

    # Reshape to flat data
    # to [BATCH_SIZE, IMAGE_SIZE / 4 * IMAGE_SIZE / 4 * 64]
    flat = tf.reshape(max_pooling2, [batch_size, -1])

    # dense layer - 384 units
    # to [BATCH_SIZE, 384]
    dense1 = tf.layers.dense(
        inputs=flat,
        units=384,
        activation=tf.nn.relu)

    # dense layer - 192 units
    # to [BATCH_SIZE, 192]
    dense2 = tf.layers.dense(
        inputs=dense1,
        units=192,
        activation=tf.nn.relu)

    # dense layer - NUM_CLASSES units
    # to [BATCH_SIZE, NUM_CLASSES]
    logits = tf.layers.dense(
        inputs=dense2,
        units=NUM_CLASSES,
        activation=tf.nn.relu)

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
        # minimise loss
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # for evaluate
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions['classes'])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
