import tensorflow as tf

def Focal_Loss(predictions, labels, alpha = 0.25, gamma = 2):
    '''
    pt =
    {
        p    , if y = 1
        1 − p, otherwise
    }

    FL(pt) = −(1 − pt)γ * log(pt)
    '''
    pt = labels * predictions + (1 - labels) * (1 - predictions) 
    focal = -alpha * (1. - pt)**gamma * tf.log(pt + 1e-9)
    focal = tf.reduce_sum(tf.abs(focal), axis = -1)
    
    return tf.reduce_mean(focal)

if __name__ == '__main__':
    import numpy as np

    BATCH_SIZE = 1
    x = np.random.rand(1000)[np.newaxis, :]
    y = np.zeros((1, 1000), dtype = np.float32)
    y[:, [0, 1, 2]] = 1.
    
    loss_op = Focal_Loss(x, y)

    sess = tf.Session()
    print(sess.run([loss_op]))