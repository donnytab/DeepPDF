import input_data
import cnn_pdf
import tensorflow as tf
import numpy as np

# TRAINING_EPOCH = 10

if __name__ == "__main__":
    # Load pdf image into array
    img_features, sample_size = input_data.load_image()

    print("size: ", sample_size)
    TRAINING_EPOCH = sample_size

    labels = np.ones(sample_size)

    # x = tf.placeholder(tf.float32, [None, cnn_pdf.PIXEL_DIMENSION_WIDTH * cnn_pdf.PIXEL_DIMENSION_HEIGHT])
    x = tf.placeholder(tf.float32, [cnn_pdf.PIXEL_DIMENSION_WIDTH * cnn_pdf.PIXEL_DIMENSION_HEIGHT])
    y = tf.placeholder(tf.int64, [1])

    # Build CNN model
    y_conv, keep_prob = cnn_pdf.cnn_pdf_model(x)

    print("yconv: ", y_conv)
    print("keep_prob: ", keep_prob)

    with tf.name_scope("loss"):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_conv)

    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope("adam_optimizer"):
        train_step = tf.train.AdadeltaOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), y)
        correct_prediction = tf.cast(correct_prediction, tf.float32)

    accuracy = tf.reduce_mean(correct_prediction)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_EPOCH):
            # _x = np.reshape(img_features[i], (cnn_pdf.PIXEL_DIMENSION_WIDTH * cnn_pdf.PIXEL_DIMENSION_HEIGHT, 1))
            _y = np.reshape(labels[i], (-1))
            train_accuracy = accuracy.eval(feed_dict={x: img_features[i], y: _y, keep_prob: 1.0})

            print("train_accuracy: ", train_accuracy)

            train_step.run(feed_dict={x: img_features[i], y: _y, keep_prob: 0.5})