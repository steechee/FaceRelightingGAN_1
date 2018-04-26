import math
import numpy as np
import tensorflow as tf
from glob import glob
import os
import h5py
import cv2

from tensorflow.python.framework import ops

from utils import *

try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

def bn(x, name, train=True):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=train,
                                        scope=name)

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.contrib.layers.xavier_initializer())
              # initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.contrib.layers.xavier_initializer())
              # initializer=tf.random_normal_initializer(stddev=stddev))

    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv


def maxpool(input, k_h, k_w, d_h, d_w, name='maxpool'):
    with tf.variable_scope(name):
        maxpool = tf.nn.max_pool(input, ksize=[1, k_h, k_w, 1], strides=[1, d_h, d_w, 1], padding='SAME')
    return maxpool


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias

def upscale(tensor, scale):
    _, h, w, _ = tensor.get_shape().as_list()
    output = tf.image.resize_nearest_neighbor(tensor, (h*scale, w*scale))
    return output


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def sigmoid_cross_entropy_with_logits(x, y):
    try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
    except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)




# addition
def hardtanh(input):
    output = tf.clip_by_value(input,-1,1)
    return output


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)


def relu(x, name="relu"):
  return tf.maximum(x, 0)


def MSECriterion(input, target):
    # loss = tf.losses.mean_squared_error(input, target)
    nelement = tf.size(input)
    loss = tf.reduce_sum(tf.square(input - target))/nelement
    return loss


def AbsCriterion(input, target):
    # loss = tf.losses.absolute_difference(input,target)
    nelement = tf.size(input)
    loss = tf.reduce_sum(tf.abs(input - target))/nelement
    return loss


def TVSelfPartialCriterion(input, mask):
    input_mask = input*mask
    x_diff = input_mask[:,:-1,:-1,:] - input_mask[:,:-1,1:,:]
    y_diff = input_mask[:,:-1,:-1,:] - input_mask[:,1:,:-1,:]
    nelement = tf.size(input)
    loss = (tf.reduce_sum(tf.abs(x_diff))+tf.reduce_sum(tf.abs(y_diff)))/nelement
    return loss


def SmoothSelfPartialCriterion(input, mask):
    input_mask = input*mask
    x_diff = input_mask[:,:-1,:-1,:] - input_mask[:,:-1,1:,:]
    y_diff = input_mask[:,:-1,:-1,:] - input_mask[:,1:,:-1,:]
    nelement = tf.size(input)
    loss = 0.5*(tf.norm(x_diff, ord = 2)+tf.norm(y_diff, ord = 2))/nelement
    return loss


def SmoothSelfCriterion(input):
    x_diff = input[:,:-1,:-1,:] - input[:,:-1,1:,:]
    y_diff = input[:,:-1,:-1,:] - input[:,1:,:-1,:]
    nelement = tf.size(input)
    loss = (tf.norm(x_diff, ord = 2)+tf.norm(y_diff, ord = 2))/nelement
    return loss


def MarginNegMSECriterion(input, target):
    margin = tf.constant(20, dtype = tf.float32)
    nelement = tf.size(input)
    loss1 = tf.reduce_sum(tf.square(input - target))/nelement
    loss2 = margin - loss1
    loss2 = tf.maximum(loss2,0)
    return loss2


def BatchWhiteShadingCriterion(input, mask):

    input_mask = input * mask
    mask_single_channel = mask[:,:,:,0] # batch*64*64
    m = tf.reduce_sum(mask_single_channel) + 1e-6 # # pixel in mask

    avg_r = tf.reduce_sum(input_mask[:,:,:,0])/m
    avg_g = tf.reduce_sum(input_mask[:,:,:,1])/m
    avg_b = tf.reduce_sum(input_mask[:,:,:,2])/m

    loss_r = 0.5*((avg_r - 0.75)**2)
    loss_g = 0.5*((avg_g - 0.75)**2)
    loss_b = 0.5*((avg_b - 0.75)**2)

    loss = loss_r + loss_g + loss_b

    return loss


def discriminatorloss(real, d_real, fake, d_fake):
    batchsize = real.shape[0]
    half =  batchsize/2

    realidx = np.zeros(real.shape[0])
    for idx in np.random.permutation(int(batchsize)):
        realidx[idx] = 1
        if np.sum(realidx) == half:
            break

    reallist = np.where(realidx == 1)
    for i in range(len(reallist[0])):
        if i == 0:
            newreal = tf.expand_dims(real[reallist[0][i],:,:,:], 0)
            newdreal = tf.expand_dims(d_real[reallist[0][i],:,:,:], 0)
        else:
            newreal = tf.concat([newreal,tf.expand_dims(real[reallist[0][i],:,:,:], 0)],axis=0) # 50 64 64 3
            newdreal = tf.concat([newdreal,tf.expand_dims(d_real[reallist[0][i],:,:,:], 0)],axis=0)

    fakelist = np.where(realidx != 1) # 50
    for i in range(len(fakelist[0])):
        if i == 0:
            newfake = tf.expand_dims(fake[fakelist[0][i],:,:,:], 0)
            newdfake = tf.expand_dims(d_fake[fakelist[0][i],:,:,:], 0)
        else:
            newfake = tf.concat([newfake,tf.expand_dims(fake[fakelist[0][i],:,:,:], 0)],axis=0)
            newdfake = tf.concat([newdfake,tf.expand_dims(d_fake[fakelist[0][i],:,:,:], 0)],axis=0)

    D_loss_real = MSECriterion(newreal, newdreal)
    D_loss_fake = MarginNegMSECriterion(newfake, newdfake)

    loss = D_loss_real + D_loss_fake

    return [D_loss_real, D_loss_fake, loss]


def SHPartialShadingRGB_bw(normal, light):
    # normal 100*64*64*3
    # light 100*10*3

    nSample = normal.shape[0] # batch_size 100
    nPixel = normal.shape[1]*normal.shape[2] # 64*64 = 4096

    Lr_bw = light[:,:,0] # 100*10
    Lr = Lr_bw[:,0:9] # 100*9
    Lg_bw = light[:,:,1]
    Lg = Lg_bw[:,0:9]
    Lb_bw = light[:,:,2]
    Lb = Lb_bw[:,0:9]

    Ns = tf.reshape(normal,[nSample, nPixel, 3]) # 100*4096*3
    N_ext = tf.ones([nSample, nPixel, 1], dtype=tf.float32) # 100*4096*1
    Ns = tf.concat([Ns, N_ext], axis=-1) # 100*4096*4

    for idx in range(nSample):
        nt = Ns[idx] # 4096*4

        mr = getmatrix(Lr[idx])
        mg = getmatrix(Lg[idx])
        mb = getmatrix(Lb[idx])

        sr = tf.matmul(nt,mr)*nt # 4096*4
        sg = tf.matmul(nt,mg)*nt
        sb = tf.matmul(nt,mb)*nt

        s1 = tf.reshape(tf.reduce_sum(sr,axis=-1)*Lr_bw[idx,9], [1,64,64]) # should be > 0 but lr_bw[idx,9] constant often < 0
        s2 = tf.reshape(tf.reduce_sum(sg,axis=-1)*Lg_bw[idx,9], [1,64,64])
        s3 = tf.reshape(tf.reduce_sum(sb,axis=-1)*Lb_bw[idx,9], [1,64,64])

        s = tf.stack([s1,s2,s3],axis=3)

        if i == 0:
            shading = s
        else:
            shading = tf.concat([S,s],axis=0)

    return shading


def SHPartialShadingRGB_bw9(normal, light):
    # normal 100*64*64*3
    # light 100*10*3

    nSample = normal.shape[0] # batch_size 100
    nPixel = normal.shape[1]*normal.shape[2] # 64*64 = 4096

    Lr_bw = light[:,:,0] # 100*10
    Lr = Lr_bw[:,0:9] # 100*9
    Lg_bw = light[:,:,1]
    Lg = Lg_bw[:,0:9]
    Lb_bw = light[:,:,2]
    Lb = Lb_bw[:,0:9]

    Ns = tf.reshape(normal,[nSample, nPixel, 3]) # 100*4096*3
    N_ext = tf.ones([nSample, nPixel, 1], dtype=tf.float32) # 100*4096*1
    Ns = tf.concat([Ns, N_ext], axis=-1) # 100*4096*4

    for idx in range(nSample):
        nt = Ns[idx] # 4096*4

        mr = getmatrix(Lr[idx])
        mg = getmatrix(Lg[idx])
        mb = getmatrix(Lb[idx])

        sr = tf.matmul(nt,mr)*nt # 4096*4
        sg = tf.matmul(nt,mg)*nt
        sb = tf.matmul(nt,mb)*nt

        s1 = tf.reshape(tf.reduce_sum(sr,axis=-1), [1,64,64]) # should be > 0 but lr_bw[idx,9] constant often < 0
        s2 = tf.reshape(tf.reduce_sum(sg,axis=-1), [1,64,64])
        s3 = tf.reshape(tf.reduce_sum(sb,axis=-1), [1,64,64])

        s = tf.stack([s1,s2,s3],axis=3)

        if i == 0:
            shading = s
        else:
            shading = tf.concat([S,s],axis=0)

    return shading


def getmatrix(L):

    c1 = 0.429043
    c2 = 0.511664
    c3 = 0.743152
    c4 = 0.886227
    c5 = 0.247708

    # print L.get_shape()
    # M = [ [c1*L9, c1*L5, c1*L8, c2*L4],
    # [c1*L5   -c1*L9   c1*L6   c2*L2
    # c1*L8   c1*L6    c3*L7   c2*L3
    # c2*L4   c2*L2    c2*L3   c4*L1 - c5*L7 ]

    e0 = tf.constant([0,0,0,0,0,0,0,0,0], dtype = tf.float32)
    e1 = tf.constant([1,0,0,0,0,0,0,0,0], dtype = tf.float32)
    e2 = tf.constant([0,1,0,0,0,0,0,0,0], dtype = tf.float32)
    e3 = tf.constant([0,0,1,0,0,0,0,0,0], dtype = tf.float32)
    e4 = tf.constant([0,0,0,1,0,0,0,0,0], dtype = tf.float32)
    e5 = tf.constant([0,0,0,0,1,0,0,0,0], dtype = tf.float32)
    e6 = tf.constant([0,0,0,0,0,1,0,0,0], dtype = tf.float32)
    e7 = tf.constant([0,0,0,0,0,0,1,0,0], dtype = tf.float32)
    e8 = tf.constant([0,0,0,0,0,0,0,1,0], dtype = tf.float32)
    e9 = tf.constant([0,0,0,0,0,0,0,0,1], dtype = tf.float32)


    L = tf.cast(tf.diag(L),dtype=tf.float32)

    M11 = c1 * tf.matmul(tf.matmul([e9,e0,e0,e0],L),tf.transpose([e9,e0,e0,e0]))
    M12 = c1 * tf.matmul(tf.matmul([e5,e0,e0,e0],L),tf.transpose([e0,e5,e0,e0]))
    M13 = c1 * tf.matmul(tf.matmul([e8,e0,e0,e0],L),tf.transpose([e0,e0,e8,e0]))
    M14 = c2 * tf.matmul(tf.matmul([e4,e0,e0,e0],L),tf.transpose([e0,e0,e0,e4]))
    M21 = c1 * tf.matmul(tf.matmul([e0,e5,e0,e0],L),tf.transpose([e5,e0,e0,e0]))
    M22 = -c1 * tf.matmul(tf.matmul([e0,e9,e0,e0],L),tf.transpose([e0,e9,e0,e0]))
    M23 = c1 * tf.matmul(tf.matmul([e0,e6,e0,e0],L),tf.transpose([e0,e0,e6,e0]))
    M24 = c2 * tf.matmul(tf.matmul([e0,e2,e0,e0],L),tf.transpose([e0,e0,e0,e2]))
    M31 = c1 * tf.matmul(tf.matmul([e0,e0,e8,e0],L),tf.transpose([e8,e0,e0,e0]))
    M32 = c1 * tf.matmul(tf.matmul([e0,e0,e6,e0],L),tf.transpose([e0,e6,e0,e0]))
    M33 = c3 * tf.matmul(tf.matmul([e0,e0,e7,e0],L),tf.transpose([e0,e0,e7,e0]))
    M34 = c2 * tf.matmul(tf.matmul([e0,e0,e3,e0],L),tf.transpose([e0,e0,e0,e3]))
    M41 = c2 * tf.matmul(tf.matmul([e0,e0,e0,e4],L),tf.transpose([e4,e0,e0,e0]))
    M42 = c2 * tf.matmul(tf.matmul([e0,e0,e0,e2],L),tf.transpose([e0,e2,e0,e0]))
    M43 = c2 * tf.matmul(tf.matmul([e0,e0,e0,e3],L),tf.transpose([e0,e0,e3,e0]))
    M44 = c4 * tf.matmul(tf.matmul([e0,e0,e0,e1],L),tf.transpose([e0,e0,e0,e1])) - c5 * tf.matmul(tf.matmul([e0,e0,e0,e7],L),tf.transpose([e0,e0,e0,e7]))

    M = M11 + M12 + M13 + M14 + M21 + M22 + M23 + M24 + M31 + M32 + M33 + M34 + M41 + M42 + M43 + M44
    M = tf.cast(M,dtype=tf.float32)

    return M



def loaddata(datapath, train, train_size):

    imglist = glob(os.path.join(datapath, '*inmc_celebA*.hdf5'))
    lightlist = glob(os.path.join(datapath, '*lrgb_celebA*.hdf5'))

    # for idx in range(1):
    for idx in range(len(imglist)):
        inmc = imglist[idx]
        f5 = h5py.File(inmc, 'r')
        imgs = f5['zx_7']

        lrgb = lightlist[idx]
        g5 = h5py.File(lrgb, 'r')
        lights = g5['zx_7']

        if idx == 0:
            imgset = imgs
            lightset = lights
        else:
            imgset = np.concatenate((imgset, imgs))
            lightset = np.concatenate((lightset, lights))

    input_all = zip(imgset, lightset)

    if not train_size:
        return input_all
    else:
        pass

    trainlist = input_all[:train_size]
    vallist = input_all[train_size:]

    if train:
        return trainlist
    else:
        return vallist



def read_data_batch(inputs):

    for idx in range(len(inputs)): # 100

        img = inputs[idx][0]

        rgb = np.expand_dims(np.stack((img[0],img[1],img[2]), axis=2), 0)
        mask = np.expand_dims(np.stack((img[6],img[6],img[6]), axis=2), 0)
        coor = np.expand_dims(np.stack((img[7],img[8],img[9]), axis=2), 0)

        normal = np.stack((img[3],img[4],img[5]), axis=2)
        normal = 2*normal -1 # 64 64 3

        for x in range(normal.shape[0]): # 64
            for y in range(normal.shape[1]): # 64
                normal[x,y] = normal[x,y]/np.sqrt(np.sum(normal[x,y]**2))

        normal = np.expand_dims(normal, 0)

        light = inputs[idx][1]
        lightrgb = np.expand_dims(np.stack((light[0],light[1],light[2]),axis=1), 0)

        if idx == 0:
            input_img = rgb
            input_normal = normal
            input_mask = mask
            input_coor = coor
            input_light = lightrgb

        else:
            input_img = np.concatenate([input_img, rgb], axis=0)
            input_normal = np.concatenate([input_normal, normal], axis=0)
            input_mask = np.concatenate([input_mask, mask], axis=0)
            input_coor = np.concatenate([input_coor, coor], axis=0)
            input_light = np.concatenate([input_light, lightrgb], axis=0)

    return [input_img, input_normal, input_mask, input_coor, input_light]



def make_grid(tensor, nrow=10, padding=2, normalize=False, scale_each=False):
    nmaps = tensor.shape[0] # # images = batch size = 100
    xmaps = min(nrow, nmaps) # nrow = 10
    ymaps = int(math.ceil(float(nmaps) / xmaps)) # ncol = 10
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding) # 64+2 = 66, 64+2 = 66
    grid = np.zeros([height * ymaps, width * xmaps, 3], dtype=np.float32) # 66*10, 66*10, 3
    k = 0
    for y in range(ymaps): # [0 9]
        for x in range(xmaps): # [0 9]
            if k >= nmaps:
                break
            h, h_width = y * height + 1, height - padding # 66*idx +1, h_width = tensor.shape[1] = 64
            w, w_width = x * width + 1, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid


# def matrixprint(input):
#     print "matrix["+("%d"%input.shape[0])+"]["+("%d"%input.shape[1])+"]"
#     for r in input.shape[0]:
#         for c in input.shage[1]:
#             print "%4.4f" %input(r,c)
#         print "\n"
