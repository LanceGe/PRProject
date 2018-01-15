import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def base_model_fn(features, labels, mode):
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

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
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

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    # The third block
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=500,
        kernel_size=[4, 4],
        activation=tf.nn.relu
    )

    # The fourth block
    conv4 = tf.layers.conv2d(
        inputs=conv3,
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
    print(onehot_labels.get_shape(), logits.get_shape())
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
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
        model_fn=base_model_fn,
        model_dir="./base_model_dropout"
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
