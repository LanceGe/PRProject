import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
s = 1
train_begin = 0
train_end = 10000
seed = 0
activation = tf.nn.relu


def SKs_model_fn(features, labels, mode):
    """ Adapted from
        https://www.tensorflow.org/tutorials/layers#building_the_cnn_mnist_classifier.
    """
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # The first block
    conv1_1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=20 * s,
        kernel_size=[3, 3],
        kernel_initializer=tf.glorot_uniform_initializer(seed=seed+0),
        activation=None
    )

    bn1_1 = tf.layers.batch_normalization(
        inputs=conv1_1,
        training=(mode == tf.estimator.ModeKeys.TRAIN)
    )

    relu1_1 = activation(bn1_1)

    conv1_2 = tf.layers.conv2d(
        inputs=relu1_1,
        filters=20 * s,
        kernel_size=[3, 3],
        kernel_initializer=tf.glorot_uniform_initializer(seed=seed+1),
        activation=None
    )

    bn1_2 = tf.layers.batch_normalization(
        inputs=conv1_2,
        training=(mode == tf.estimator.ModeKeys.TRAIN)
    )

    relu1_2 = activation(bn1_2)

    pool1 = tf.layers.max_pooling2d(
        inputs=relu1_2,
        pool_size=[2, 2],
        strides=2
    )

    # The second block
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=50 * s,
        kernel_size=[5, 5],
        kernel_initializer=tf.glorot_uniform_initializer(seed=seed+2),
        activation=None
    )

    bn2 = tf.layers.batch_normalization(
        inputs=conv2,
        training=(mode == tf.estimator.ModeKeys.TRAIN)
    )

    relu2 = activation(bn2)

    pool2 = tf.layers.max_pooling2d(
        inputs=relu2,
        pool_size=[2, 2],
        strides=2
    )

    # The third block
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=500 * s,
        kernel_size=[4, 4],
        kernel_initializer=tf.glorot_uniform_initializer(seed=seed+3),
        activation=None
    )

    bn3 = tf.layers.batch_normalization(
        inputs=conv3,
        training=(mode == tf.estimator.ModeKeys.TRAIN)
    )

    relu3 = activation(bn3)

    # The fourth block
    conv4 = tf.layers.conv2d(
        inputs=relu3,
        filters=10,
        kernel_size=[1, 1],
        kernel_initializer=tf.glorot_uniform_initializer(seed=seed+4),
        activation=None
    )

    logits = tf.reshape(conv4, [-1, 10])

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probability": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        # add moving average to the training set
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step()
            )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_arg):
    from utils import mnist_dataset
    import numpy as np

    train_data = mnist_dataset["train_image"][train_begin:train_end].astype(np.float32)
    train_labels = mnist_dataset["train_label"][train_begin:train_end].astype(np.int32)
    eval_data = mnist_dataset["test_image"].astype(np.float32)
    eval_labels = mnist_dataset["test_label"].astype(np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=SKs_model_fn,
        model_dir="./SKs_model_" + "_".join(sys.argv[1:])
    )

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log,
        every_n_iter=50
    )

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=20,
        shuffle=True,
        seed=0
    )

    mnist_classifier.train(
        input_fn=train_input_fn,
        hooks=[logging_hook]
    )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == '__main__':
    import sys
    import numpy as np
    s = float(sys.argv[1])
    train_begin = int(sys.argv[2])
    train_end = int(sys.argv[3])
    seed = int(sys.argv[4])
    if sys.argv[5] == "sigmoid":
        activation = tf.nn.sigmoid
    tf.set_random_seed(seed)
    tf.app.run()
