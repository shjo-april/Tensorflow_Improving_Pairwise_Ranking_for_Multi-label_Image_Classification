import numpy as np
import tensorflow as tf

def Log_Sum_Exp_Pairwise_Loss(predictions, labels, size = 1):
    loss_op = 0.0

    for i in range(size):
        positive = tf.boolean_mask(predictions[i], tf.math.equal(labels[i], 1.))
        negative = tf.boolean_mask(predictions[i], tf.math.equal(labels[i], 0.))

        exp_sub = tf.exp(negative[:, tf.newaxis] - positive[tf.newaxis, :])
        exp_sum = tf.reduce_sum(exp_sub)
        
        loss_op += tf.log(1 + exp_sum)
    
    return loss_op / size

if __name__ == '__main__':
    predictions = tf.placeholder(tf.float32, [None, 1000])
    labels = tf.placeholder(tf.float32, [None, 1000])

    loss_op = Log_Sum_Exp_Pairwise_Loss(predictions, labels)
    print(loss_op)

    sess = tf.Session()
    
    pred_vector = np.random.rand(1 * 1000)
    pred_vector = np.reshape(pred_vector, (1, 1000))

    label_vector = np.zeros((1, 1000))
    label_vector[:, :10] = 1

    loss = sess.run(loss_op, feed_dict = {predictions : pred_vector, labels : label_vector})
    print(loss)
