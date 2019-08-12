import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def test(sess,gen_sample,noise_dim,gen_input):
    for i in range(10):
        z = np.random.uniform(-1., 1., size=[4, noise_dim])
        g = sess.run([gen_sample], feed_dict={gen_input: z})
        # Reverse colours for better display
