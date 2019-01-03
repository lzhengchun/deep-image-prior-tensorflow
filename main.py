
import tensorflow as tf
import numpy as np
import glob, skimage.measure, skimage.io, imageio, os, shutil, time

def save2img(d_img, fn):
    if len(d_img.shape) == 2:
        _min, _max = d_img.min(), d_img.max()
        img = (d_img - _min) * 255. / (_max - _min)
    else:
        img = np.zeros((d_img.shape))
        for c in range(d_img.shape[-1]):
            _min, _max = d_img[:, :, c].min(), d_img[:, :, c].max()
            img[:, :, c] = (d_img[:, :, c] - _min) * 255. / (_max - _min)

    img = img.astype('uint8')
    imageio.imwrite(fn, img)

def unet_conv_block(inputs, nch):
    outputs = tf.layers.conv2d(inputs, nch, 3, padding='same', activation=tf.nn.relu, \
                               kernel_initializer=tf.contrib.layers.xavier_initializer())

    outputs = tf.layers.conv2d(outputs, nch, 3, padding='same', activation=tf.nn.relu, \
                               kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    return outputs
    
def unet_upsample(inputs, nlayers=4):
    outputs = {}
    outputs[1] = unet_conv_block(inputs, 32)
    for layer in range(2, nlayers+1):
        _tmp = tf.layers.max_pooling2d(outputs[layer-1], pool_size=2, strides=2, padding='same')
        outputs[layer] = unet_conv_block(_tmp, _tmp.shape[-1]*2) 
        
    # intermediate layers
    _tmp = tf.layers.max_pooling2d(outputs[nlayers], pool_size=2, strides=2, padding='same')
    _tmp = tf.layers.conv2d(_tmp, _tmp.shape[3], 3, padding='same', \
                            activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
    for layer, nch in zip(range(1, nlayers+1), (128, 64, 32, 32)):
        _tmp = tf.image.resize_images(_tmp, (2*_tmp.shape[1], 2*_tmp.shape[2]))
        _tmp = tf.concat([outputs[nlayers-layer+1], _tmp], 3)
        _tmp = unet_conv_block(_tmp, nch)

    _tmp = tf.layers.conv2d(_tmp, 16, 1, padding='valid', \
                            kernel_initializer=tf.contrib.layers.xavier_initializer(), \
                            activation=tf.nn.relu)
    _tmp = tf.layers.conv2d(_tmp, inputs.shape[-1],  1, padding='valid', \
                            kernel_initializer=tf.contrib.layers.xavier_initializer(), \
                            activation=tf.nn.tanh)
    return _tmp    
        
def main():
    it_out_path = 'out'
    if os.path.isdir(it_out_path): 
        shutil.rmtree(it_out_path)
    os.mkdir(it_out_path)

    noisy_img = np.expand_dims(skimage.io.imread('img-prior-in/snail.jpg')[2:258]/255. - 1., 0) # norm to [0, 1]

    if len(noisy_img.shape) < 4: noisy_img = np.expand_dims(noisy_img, 3)
    img_w, img_h, img_c = noisy_img.shape[1], noisy_img.shape[2], noisy_img.shape[3]

    batch_size = 1
    tf.reset_default_graph()
    tnsr_img_noisy  = tf.placeholder(dtype=tf.float32, shape=[batch_size, img_w, img_h, img_c])
    tnsr_rnd_in     = tf.placeholder(dtype=tf.float32, shape=[batch_size, img_w, img_h, img_c])
    with tf.variable_scope('generator') as scope:
        img_gen = unet_upsample(tnsr_rnd_in, nlayers=4)
    mse_loss= tf.reduce_sum(tf.squared_difference(tnsr_img_noisy, img_gen)) / (np.prod(noisy_img.shape))

    gen_params   = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    gen_train_op = tf.train.AdamOptimizer().minimize(mse_loss, var_list=gen_params)

    rnd_in = np.random.rand(batch_size, img_w, img_h, img_c)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for it in xrange(8000):
            time_cnt = time.time()
            _img_gen, _mse_loss, _ = sess.run([img_gen, mse_loss, gen_train_op], feed_dict={tnsr_rnd_in:rnd_in, tnsr_img_noisy:noisy_img})

            if it % 500 == 0:
                save2img(_img_gen[0], "%s/denoised-it%03d.png" % (it_out_path, it))
                if it==0: save2img(noisy_img[0], "%s/noisy-in.png" % (it_out_path, ))
                print "it:%03d, mse-loss:%.3f, it_Telapse: %.3fs" % (it, _mse_loss, time.time() - time_cnt)
        
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
