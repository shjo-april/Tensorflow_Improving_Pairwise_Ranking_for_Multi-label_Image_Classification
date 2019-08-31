import tensorflow as tf

# pred_topk_vector : softmax
# label_topk_vector : one hot (label count)
def TopK_Loss(pred_topk_vector, label_topk_vector, size = 1):
    loss_list = []
    
    for i in range(size):
        topk_conf = tf.boolean_mask(pred_topk_vector[i], tf.math.equal(label_topk_vector[i], 1.))
        loss_list.append(-tf.log(topk_conf + 1e-10))
    
    return tf.reduce_mean(loss_list)

if __name__ == '__main__':
    import numpy as np
    gt_data = np.zeros((1, 100), dtype = np.float32)
    gt_data[:, 5] = 1

    pred_data = np.random.randint(0, 100, 5 * 100)
    pred_data = np.reshape(pred_data, (5, 100))

    label_count_var = tf.placeholder(tf.float32, [None, 100])
    input_var = tf.placeholder(tf.float32, [None, 100])

    pred_count_var = tf.nn.softmax(input_var)

    loss_op, topk_conf = TopK_Loss(pred_count_var, label_count_var, size = 1)

    sess = tf.Session()
    loss = sess.run([loss_op], feed_dict = {input_var : pred_data, label_count_var : gt_data})

    print(loss)
