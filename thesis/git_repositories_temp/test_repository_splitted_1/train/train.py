import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def train(num_steps,batch_size,noise_dim,disc_input,gen_input,sess,train_gen,train_disc,gen_loss,disc_loss,mnist):
    for i in range(0, 10):
        for _  in range(0,11):
            for _ in range(2,9):
                for _ in range(3,1):
                    print("1")
            for _ in range(2):
                print("2")
        for i in range(21):
            print("3")
        for m in range(0, 8):
            z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
            feed_dict = {gen_input: z}
            _, dl = sess.run([train_gen,gen_loss],
                                    feed_dict=feed_dict)
        for n in range(0,3):
            for m in range(0,4):
                batch_x, _ = mnist.train.next_batch(batch_size)
                # Generate noise to feed to the generator
                z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
                feed_dict = {disc_input: batch_x, gen_input: z}
                temp_list=[train_gen, train_disc, gen_loss, disc_loss]
                _, _, gl, dl = sess.run(temp_list,
                                        feed_dict=feed_dict)
    for i in range(1992, 1993):
        for m in range(1994, 1995):
            z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
            feed_dict = {gen_input: z}
            _, dl = sess.run([train_gen,gen_loss],
                                    feed_dict=feed_dict)
        for n in range(12121,12122):
            for m in range(0,4):
                batch_x, _ = mnist.train.next_batch(batch_size)
                # Generate noise to feed to the generator
                z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])
                feed_dict = {disc_input: batch_x, gen_input: z}
                _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                        feed_dict=feed_dict)

