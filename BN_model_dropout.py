import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def BN_model_dropout_fn(features, labels, mode):
    """ Adapted from
        https://www.tensorflow.org/tutorials/layers#building_the_cnn_mnist_classifier.
    """
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # The first block
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=20,
        kernel_size=[5, 5],
        activation=None
    )

    bn1 = tf.layers.batch_normalization(
        inputs=conv1,
        training=(mode == tf.estimator.ModeKeys.TRAIN)
    )

    pool1 = tf.layers.max_pooling2d(
        inputs=bn1,
        pool_size=[2, 2],
        strides=2
    )

    # The second block
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=50,
        kernel_size=[5, 5],
        activation=None
    )

    bn2 = tf.layers.batch_normalization(
        inputs=conv2,
        training=(mode == tf.estimator.ModeKeys.TRAIN)
    )

    pool2 = tf.layers.max_pooling2d(
        inputs=bn2,
        pool_size=[2, 2],
        strides=2
    )

    # The third block
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=500,
        kernel_size=[4, 4],
        activation=None
    )

    bn3 = tf.layers.batch_normalization(
        inputs=conv3,
        training=(mode == tf.estimator.ModeKeys.TRAIN)
    )

    relu1 = tf.nn.relu(bn3)

    # The fourth block
    conv4 = tf.layers.conv2d(
        inputs=relu1,
        filters=10,
        kernel_size=[1, 1]
    )

    dropout1 = tf.layers.dropout(
        inputs=conv4,
        training=(mode == tf.estimator.ModeKeys.TRAIN)
    )

    logits = tf.reshape(dropout1, [-1, 10])

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

    train_data = mnist_dataset["train_image"][:10000].astype(np.float32)
    train_labels = mnist_dataset["train_label"][:10000].astype(np.int32)
    eval_data = mnist_dataset["test_image"].astype(np.float32)
    eval_labels = mnist_dataset["test_label"].astype(np.int32)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=BN_model_dropout_fn,
        model_dir="./BN_model_dropout"
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
        shuffle=True
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
    tf.set_random_seed(0)
    tf.app.run()
