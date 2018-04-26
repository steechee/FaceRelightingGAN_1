import os, sys, time, math
from glob import glob
import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
from tqdm import tqdm, trange
import cv2
from ops import *
import datetime
import scipy.io as sio
import imageio
import scipy.misc

class BeU(object):
    def __init__(self, sess, input_h=64, input_w=64, output_h=64, output_w=64, batch_size=100, log_dir=None, sum_dir=None):

        self.sess = sess
        self.batch_size = batch_size
        self.input_h = input_h
        self.input_w = input_w
        self.output_h = output_h
        self.output_w = output_w

        self.log_dir = log_dir
        self.sum_dir = sum_dir

        self.input_img = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_h, self.input_w, 3], name='input_img')
        self.input_normal = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_h, self.input_w, 3], name='input_normal')
        self.input_mask = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_h, self.input_w, 3], name='input_mask')
        self.input_coor = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_h, self.input_w, 3], name='input_coor')
        self.input_light = tf.placeholder(tf.float32, shape=[self.batch_size, 9, 3], name='light_gt')

        self.build_model()

    def build_model(self):

        self.input = self.input_img # [0 1]
        self.mask_gt = self.input_mask # [0 1]
        self.normal_gt = self.input_normal # [-1 1]
        self.light_gt = self.input_light # [0 1]

        self.z, self.e = self.encoder(self.input, 'enc', reuse = False)

        # fix bg, shading, albedo, light, mask
        self.bg = hardtanh(self.input * (1-self.mask_gt))
        self.shading_gt = hardtanh(SHPartialShadingRGB_bw9(self.normal_gt, self.light_gt)) # 100 batch *64*64*3
        self.albedo = hardtanh(self.input/(self.shading_gt + 1e-6))
        self.mask = self.mask_gt
        self.light = self.light_gt

        # normal control test
        # self.bg = self.decoderBM(self.z[4], self.e, 'dec_B', reuse = False) # 100, 64, 64, 3
        # self.mask = self.decoderM(self.z[5], self.e, 'dec_M', reuse = False) #  100, 64, 64, 3
        # self.light = self.decoderL(self.z[3], 'dec_L', reuse = False) # 100, 10, 3
        self.normal, self.normalnm = self.decoderN(self.z[2], 'dec_N', reuse = False) # 100, 64, 64, 3 / 100, 64, 64, 1
        # self.albedo = self.decoderA(self.z[1], 'dec_A', reuse = False) # 100, 64, 64, 3

        self.shading, self.fg, self.output = self.decoderO(self.bg, self.mask, self.light, self.normal, self.albedo, 'dec_O', reuse = False)
        # self.shading, self.fg, self.output, self.fg2 = self.decoderO(self.bg, self.mask, self.light, self.normal, self.albedo, 'dec_O', reuse = False)

        # self.input_real = self.input
        self.output_real = self.discriminator(self.input, 'disc', reuse = False)
        # self.input_gene = self.output
        self.output_gene = self.discriminator(self.output, 'disc', reuse = True)

        # loss
        # self.D_loss_real = tf.losses.mean_squared_error(self.input,self.output_real)
        # self.D_loss_fake = MarginNegMSECriterion(self.output, self.output_gene)
        self.D_loss_real, self.D_loss_fake, self.D_loss = discriminatorloss(self.input,self.output_real, self.output, self.output_gene)
        # self.D_loss = self.D_loss_real + self.D_loss_fake

        self.gt_Nnm = tf.ones(tf.shape(self.normalnm)) # normal normalized check
        # self.output_adv = self.discriminator(self.output, 'disc', reuse = True)

        self.f_synth = 4*tf.losses.mean_squared_error(self.mask_gt*self.input,self.mask_gt*self.fg)
        self.f_A = 0.2*TVSelfPartialCriterion(self.albedo,self.mask_gt)
        self.f_N = 0.2*tf.losses.mean_squared_error(self.normal_gt,self.normal) + 0.2*SmoothSelfPartialCriterion(self.normal,self.mask_gt)
        self.f_Nnm = 5*tf.losses.mean_squared_error(self.gt_Nnm,self.normalnm)
        self.f_L = 5*tf.losses.mean_squared_error(self.light_gt,self.light[:,0:9]) # input, target are both 100 9 3
        # self.f_S = BatchWhiteShadingCriterion(self.shading,self.mask_gt) + SmoothSelfCriterion(self.shading)
        self.f_S = BatchWhiteShadingCriterion(self.shading_gt,self.mask_gt) + SmoothSelfCriterion(self.shading_gt)

        # self.f_B = tf.losses.mean_squared_error(self.bg,self.bg)
        self.f_M = 5*tf.losses.absolute_difference(self.mask_gt,self.mask) + 2*SmoothSelfCriterion(self.mask)

        # self.f_adv = tf.losses.mean_squared_error(self.output, self.output_adv)
        self.f_adv = tf.losses.mean_squared_error(self.output, self.output_gene) # generator adversarial loss fake real
        self.f_final_recon = 2*tf.losses.absolute_difference(self.input, self.output)
        self.f_final = self.f_adv + self.f_final_recon

        # self.U_loss = self.f_synth + self.f_A + self.f_N + self.f_Nnm + self.f_L + self.f_S + self.f_B + self.f_M + self.f_final
        self.U_loss = self.f_synth + self.f_A + self.f_N + self.f_Nnm + self.f_L + self.f_S + self.f_M + self.f_final


        ### Summary
        # image summary
        # discriminator update: input1, output1_f, outputs_gene
        self.input_sum = image_summary("input", tf.clip_by_value((self.input)*255, 0, 255))
        self.output_sum = image_summary("output", tf.clip_by_value((self.output)*255, 0, 255))
        self.output_gene_sum = image_summary("output_gene", tf.clip_by_value((self.output_real)*255, 0, 255))

        # encoder-decoder update: input3, bg3, mask3, light3, normal3, albedo3, shading3, fg3, output3
        self.bg_sum = image_summary("bg", tf.clip_by_value((self.bg)*255, 0, 255))
        self.mask_sum = image_summary("mask", tf.clip_by_value((self.mask)*255, 0, 255))
        self.normal_sum = image_summary("normal", tf.clip_by_value((self.normal+1)*127.5, 0, 255)) # return to  [0 1] -> [0 255]
        # self.normal_sum = image_summary("normal", tf.clip_by_value((self.normal)*255, 0, 255))
        self.albedo_sum = image_summary("albedo", tf.clip_by_value((self.albedo)*255, 0, 255))
        self.shading_gt_sum = image_summary("shading", tf.clip_by_value((self.shading_gt+1)*127.5, 0, 255))
        self.shading_sum = image_summary("shading", tf.clip_by_value((self.shading+1)*127.5, 0, 255))
        # self.shading_sum = image_summary("shading", tf.clip_by_value((self.shading+1)*127.5, 0, 255))
        self.fg_sum = image_summary("fg", tf.clip_by_value(self.fg*225, 0, 255))
        # self.fg2_sum = image_summary("fg2", tf.clip_by_value(self.fg2*225, 0, 255))
        # self.mask_sum = image_summary("mask", tf.clip_by_value(tf.pow(10., ((self.mask+1)/2-1)), 0, 1))

        # scalar summary
        # light coefficient
        self.light_gt_sum = tf.summary.tensor_summary("light_gt", self.light_gt)
        self.light_sum = tf.summary.tensor_summary("light", self.light)
        # losses
        # discriminator update
        self.D_loss_sum = scalar_summary("D_loss", self.D_loss)
        self.D_loss_real_sum = scalar_summary("D_loss_real", self.D_loss_real)
        self.D_loss_fake_sum = scalar_summary("D_loss_fake", self.D_loss_fake)
        # encoder-decoder update
        self.f_synth_sum = scalar_summary("f_synth", self.f_synth)
        self.f_A_sum = scalar_summary("f_A", self.f_A)
        self.f_N_sum = scalar_summary("f_N", self.f_N)
        self.f_Nnm_sum = scalar_summary("f_Nnm", self.f_Nnm)
        self.f_L_sum = scalar_summary("f_L", self.f_L)
        self.f_S_sum = scalar_summary("f_S", self.f_S)
        # self.f_B_sum = scalar_summary("f_B", self.f_B)
        self.f_M_sum = scalar_summary("f_M", self.f_M)
        self.f_adv_sum = scalar_summary("f_adv", self.f_adv)
        self.f_final_recon_sum = scalar_summary("f_final_recon", self.f_final_recon)
        self.U_loss_sum = scalar_summary("U_loss", self.U_loss)

        self.saver = tf.train.Saver()
        t_vars = tf.trainable_variables()

        self.D_vars = [var for var in t_vars if 'disc' in var.name]
        self.U_vars = [var for var in t_vars if 'enc' in var.name] + [var for var in t_vars if 'dec' in var.name]

        self.saver = tf.train.Saver()


    def train(self, config):

        # global_step = tf.Variable(0, trainable=False)
        saver = tf.train.Saver(max_to_keep=1000)

        D_opt = tf.train.AdamOptimizer(config.D_lr).minimize(self.D_loss, var_list = self.D_vars)
        U_opt = tf.train.AdamOptimizer(config.U_lr).minimize(self.U_loss, var_list = self.U_vars)

        # with tf.control_dependencies([D_opt, U_opt]):
            # ????
            # self.k_update = tf.assign(self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

        self.sess.run(tf.global_variables_initializer())

        self.D_image_sum = merge_summary([self.input_sum, self.output_sum, self.output_gene_sum])
        self.U_image_sum = merge_summary([self.input_sum, self.bg_sum, self.mask_sum, self.normal_sum, self.albedo_sum, self.shading_sum, self.shading_gt_sum, self.fg_sum, self.output_sum])
        self.L_sum = merge_summary([self.light_gt_sum, self.light_sum])
        self.D_sum = merge_summary([self.D_loss_sum, self.D_loss_real_sum, self.D_loss_fake_sum])
        self.U_sum = merge_summary([self.f_synth_sum, self.f_A_sum, self.f_N_sum, self.f_Nnm_sum, self.f_L_sum, self.f_S_sum, self.f_M_sum, self.f_adv_sum, self.f_final_recon_sum, self.U_loss_sum])

        self.writer = SummaryWriter(self.sum_dir, self.sess.graph)

        trainlist = loaddata('celeba/', train = True, train_size = 45000) # [45000*64*64*10 / 45000*9*3]
        # trainlist = loaddata('celeba/', train = True, train_size = 5000) # [45000*64*64*10 / 45000*9*3]
        num_batch = len(trainlist)//self.batch_size
        # num_batch = 45000/100

        ckpt = tf.train.get_checkpoint_state(self.log_dir)

        if ckpt:
            print('[*] Loaded '+ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,ckpt.model_checkpoint_path)
        else:
            print(" [!] Load failed...")

        start_time = time.time()
        epoch = config.last_epoch
        batchidx = 0

        print '=== gan train start! ==='

        while epoch < config.gan_epoch:
            # dataflow.reset_state()

            for randomidx in np.random.permutation(num_batch):

                # data = read_data_batch(trainlist[randomidx:randomidx+100])
                data = read_data_batch(trainlist[randomidx:randomidx+100])
                # print data[0]
                batchidx += 1
                # print data[0][5]

                feed = dict(zip([self.input_img, self.input_normal, self.input_mask, self.input_coor, self.input_light], data))

                ''' D update'''
                D_image_summary, D_summary, _ = self.sess.run([self.D_image_sum, self.D_sum, D_opt], feed_dict=feed)
                # D_image_summary, D_summary, _ = self.sess.run([self.D_image_sum, self.D_sum, D_opt], feed_dict=feed)
                # D_image_summary, D_summary, _ = self.sess.run([self.D_image_sum, self.D_sum, D_opt], feed_dict=feed)
                # D_image_summary, D_summary, _ = self.sess.run([self.D_image_sum, self.D_sum, D_opt], feed_dict=feed)
                # D_image_summary, D_summary, _ = self.sess.run([self.D_image_sum, self.D_sum, D_opt], feed_dict=feed)

                ''' E/D update'''
                disout, U_image_summary, U_summary, L_summary, _ = self.sess.run([self.output_real, self.U_image_sum, self.U_sum, self.L_sum, U_opt], feed_dict=feed)

                # testinput, testnormal, testshading, testalbedo, testoutput = self.sess.run([], feed_dict=feed)

                errDr = self.D_loss_real.eval(feed_dict=feed, session=self.sess)
                errDf = self.D_loss_fake.eval(feed_dict=feed, session=self.sess)
                errD = self.D_loss.eval(feed_dict=feed, session=self.sess)
                errU = self.U_loss.eval(feed_dict=feed, session=self.sess)

                inputrecon = self.output_real.eval(feed_dict=feed, session=self.sess)
                dmax = np.amax(inputrecon)
                dmin = np.amin(inputrecon)
                dmean = np.mean(inputrecon)

                # return [z, zA, zN, zL, zB, zM], [mp1, mp2, mp3]
                # z = self.z[0][0].eval(feed_dict=feed, session=self.sess)
                # print "z:"
                # print z
                # za = self.z[1][0].eval(feed_dict=feed, session=self.sess)
                # print "z albedo:"
                # print za
                # zn = self.z[2][0].eval(feed_dict=feed, session=self.sess)
                # print "z normal:"
                # print zn
                # zl = self.z[3][0].eval(feed_dict=feed, session=self.sess)
                # print "z light:"
                # print zl
                # zb = self.z[4][0].eval(feed_dict=feed, session=self.sess)
                # print "z background:"
                # print zb
                # zm = self.z[5][0].eval(feed_dict=feed, session=self.sess)
                # print "z mask:"
                # print zm


                # inputlight = self.light_gt[0].eval(feed_dict=feed, session=self.sess) # 9 3
                # estimatedlight = self.light[0].eval(feed_dict=feed, session=self.sess) # 10 3
                errL = self.f_L.eval(feed_dict=feed, session=self.sess)
                # print "ground truth light:"
                # print inputlight
                # print "estimated light:"
                # print estimatedlight

                if not batchidx%5: # save summary for each 20 batch
                    print "Epoch: [%2d] [%4d/%4d] time:%4.4f, D_lossr:[%.4f], D_lossf:[%.4f], D_loss:[%.4f], U_loss:[%.4f]"%(epoch, batchidx, num_batch, time.time()-start_time, errDr, errDf, errD, errU)
                    print "light loss: [%4.4f], discriminator max: [%4.4f], min: [%4.4f], avg: [%4.4f]"%(errL, dmax, dmin, dmean)
                    # scipy.misc.imsave('discriminatoroutputtest_0424.png',disout[5])

                    # print "light_gt: [%4d] \n estimated_light: [%4d]"%(inputlight, estimatedlight)
                    # matrixprint(inputlight)
                    # print np.matrix(inputlight)
                    # matrixprint(estimatedlight)
                    # print "light_gt: [%4d] \n estimated_light: [%4d]"%(inputlight, estimatedlight)
                    # print "Epoch: [%2d] [%4d/%4d] time:%4.4f, D_loss:[%.4f], U_loss:[%.4f]"%(epoch, batchidx, num_batch, time.time()-start_time, errD, errU)
                    # print "Epoch: [%2d] [%4d/%4d] time:%4.4f"%(epoch, batchidx, num_batch, time.time()-start_time)
                    self.writer.add_summary(D_image_summary, epoch)
                    self.writer.add_summary(U_image_summary, epoch)
                    self.writer.add_summary(D_summary, epoch)
                    self.writer.add_summary(U_summary, epoch)
                    self.writer.add_summary(L_summary, epoch)


                else:
                    pass

                if batchidx == num_batch:
                    batchidx = 0
                    self.saver.save(self.sess,"%s/model_%d.ckpt"%(self.log_dir, epoch))
                    epoch += 1
                    # imgidx = 0
                    if epoch == config.gan_epoch:
                        break
                else:
                    pass



    def test(self, istrain):
        saver = tf.train.Saver(max_to_keep=1000)
        ckpt = tf.train.get_checkpoint_state(self.log_dir)

        if ckpt:
            print('[*] Loaded '+ckpt.model_checkpoint_path)
            self.saver.restore(self.sess,ckpt.model_checkpoint_path)
        else:
            print(" [!] Load failed...")
            sys.exit()

        testlist = loaddata('celeba/', train = False, train_size = 45000) # 50762-45000 = 5762
        # testlist = loaddata('celeba/', train = False, train_size = 8000) # 50762-45000 = 5762
        test_num_batch = len(testlist)//self.batch_size # 57

        now = datetime.datetime.now()
        result_dir = 'test_result/%s'%(now.strftime('%m%d'))

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        loss = []
        batchidx = 0
        imgidx = 0

        for randomidx in np.random.permutation(test_num_batch):

            data = read_data_batch(testlist[randomidx:randomidx+100])
            batchidx += 1
            print batchidx
            feed = dict(zip([self.input_img, self.input_normal, self.input_mask, self.input_coor, self.input_light], data))

            # if not batchidx%10: # save summary for each 10 batch # 5 times
            if not batchidx%5: # save summary for each 10 batch # 5 times
                imgidx += 1
                inputimg, bgimg, maskimg, lightcoeff, normalimg, albedoimg, shadingimg, fgimg, outputimg, inputrecon, outputrecon  = self.sess.run([self.input, self.bg, self.mask, self.light, self.normal, self.albedo, self.shading, self.fg, self.output, self.output_real, self.output_gene], feed_dict=feed) # 100 64 64 3


                inputrecon = inputrecon/np.amax(inputrecon)
                outputrecon = outputrecon/np.amax(outputrecon)

                inputimggrid = make_grid(inputimg)
                bgimggrid = make_grid(bgimg)
                maskimggrid = make_grid(maskimg)
                # normalimggrid = make_grid(normalimg)
                normalimggrid2 = make_grid((normalimg+1)/(2+1e-6))
                albedoimggrid = make_grid(albedoimg)
                shadingimggrid = make_grid(shadingimg)
                # shadingimggrid2 = make_grid((shadingimg+1)/(2+1e-6))
                fgimggrid = make_grid(fgimg)
                # fgimggrid1_2 = make_grid(fgimg+1/(2+1e-6))
                # fgimggrid2 = make_grid(fgimg2)
                # fgimggrid2_2 = make_grid(fgimg2+1/(2+1e-6))
                outputimggrid = make_grid(outputimg)
                inputrecongrid = make_grid(inputrecon)
                outputrecongrid = make_grid(outputrecon)

                if len(str(imgidx)) == 1:
                    testname = '00'+str(imgidx)
                elif len(str(imgidx)) == 2:
                    testname = '0'+str(imgidx)

                outname = "%s/%s.png"%(result_dir, testname)

                # scipy.misc.imsave(outname,inputimggrid)
                scipy.misc.imsave(result_dir+'/'+testname+'_0_input.png',inputimggrid)
                scipy.misc.imsave(result_dir+'/'+testname+'_1_bg.png',bgimggrid)
                scipy.misc.imsave(result_dir+'/'+testname+'_2_mask.png',maskimggrid)
                # scipy.misc.imsave(result_dir+'/'+testname+'_3_normal.png',normalimggrid)
                scipy.misc.imsave(result_dir+'/'+testname+'_3_normal.png',normalimggrid2)
                scipy.misc.imsave(result_dir+'/'+testname+'_4_albedo.png',albedoimggrid)
                scipy.misc.imsave(result_dir+'/'+testname+'_5_shading.png',shadingimggrid)
                # scipy.misc.imsave(result_dir+'/'+testname+'_5_shading_2.png',shadingimggrid2)
                scipy.misc.imsave(result_dir+'/'+testname+'_6_fg.png',fgimggrid)
                # scipy.misc.imsave(result_dir+'/'+testname+'_62_fg.png',fgimggrid1_2)
                # scipy.misc.imsave(result_dir+'/'+testname+'_63_fg.png',fgimggrid2)
                # scipy.misc.imsave(result_dir+'/'+testname+'_64_fg.png',fgimggrid2_2)
                scipy.misc.imsave(result_dir+'/'+testname+'_7_output.png',outputimggrid)
                scipy.misc.imsave(result_dir+'/'+testname+'_8_inputrecon.png',inputrecongrid)
                scipy.misc.imsave(result_dir+'/'+testname+'_9_outputrecon.png',outputrecongrid)


            # lossname = "%s/loss.npy"%(result_dir)
            # loss = np.array(loss)
            # np.save(lossname, loss)



    def encoder(self, input, name, reuse=False):
        with tf.variable_scope(name):

            net = conv2d(input, 96, 5, 5, 1, 1, name='%s_conv1'%name) # [0 1]
            net = tf.maximum(relu(conv2d(net, 96, 2, 2, 2, 2, name='%s_maxpool1'%name)),1e-6) # [0 1]
            mp1 = net

            net = conv2d(net, 48, 5, 5, 1, 1, name='%s_conv2'%name)
            net = tf.maximum(relu(conv2d(net, 48, 2, 2, 2, 2, name='%s_maxpool2'%name)),1e-6) # [0 1]
            mp2 = net

            net = conv2d(net, 24, 5, 5, 1, 1, name='%s_conv3'%name)
            net = tf.maximum(relu(conv2d(net, 24, 2, 2, 2, 2, name='%s_maxpool3'%name)),1e-6) # [0 1]
            mp3 = net

            net = tf.reshape(net,[input.shape[0], -1])
            net = tf.contrib.layers.fully_connected(net, 128, activation_fn=None)

            z = tf.sigmoid(net) # [0 1]

            zA = tf.contrib.layers.fully_connected(z, 128, activation_fn=None)
            zN = tf.contrib.layers.fully_connected(z, 128, activation_fn=None)
            zL = tf.contrib.layers.fully_connected(z, 9, activation_fn=None)
            zB = tf.contrib.layers.fully_connected(z, 128, activation_fn=None)
            zM = tf.contrib.layers.fully_connected(z, 128, activation_fn=None)

        return [z, zA, zN, zL, zB, zM], [mp1, mp2, mp3]


    # decoder for B and M
    def decoderBM(self, latent, en, name, reuse=False):
        with tf.variable_scope(name):

            net = tf.maximum(tf.contrib.layers.fully_connected(latent, 24*8*8, activation_fn=None),1e-6)

            net = tf.reshape(net,[self.batch_size, 8, 8, 24])

            net = tf.concat([net, en[2]], 3) # skip connection
            net = deconv2d(net, [self.batch_size, 16, 16, 24], 2, 2, 2, 2, name='re_deconv1') # maxunpool
            net = tf.maximum(relu(conv2d(net, 48, 5, 5, 1, 1, name='re_conv1')),1e-6) # conv2d

            net = tf.concat([net, en[1]], 3)
            net = deconv2d(net, [self.batch_size, 32, 32, 48], 2, 2, 2, 2, name='re_deconv2')
            net = tf.maximum(relu(conv2d(net, 96, 5, 5, 1, 1, name='re_conv2')),1e-6)

            net = tf.concat([net, en[0]], 3)
            net = deconv2d(net, [self.batch_size, 64, 64, 96], 2, 2, 2, 2, name='re_deconv3')
            net = tf.maximum(relu(conv2d(net, 96, 5, 5, 1, 1, name='re_conv3')),1e-6)

            net = hardtanh(conv2d(net, 3, 3, 3, 1, 1, name='re_conv4'))

        return net

    def decoderM(self, latent, en, name, reuse=False):
        with tf.variable_scope(name):

            net = tf.maximum(tf.contrib.layers.fully_connected(latent, 24*8*8, activation_fn=None),1e-6)

            net = tf.reshape(net,[self.batch_size, 8, 8, 24])

            net = tf.concat([net, en[2]], 3) # skip connection
            net = deconv2d(net, [self.batch_size, 16, 16, 24], 2, 2, 2, 2, name='M_deconv1') # maxunpool
            net = tf.maximum(relu(conv2d(net, 48, 5, 5, 1, 1, name='M_conv1')),1e-6) # conv2d

            net = tf.concat([net, en[1]], 3)
            net = deconv2d(net, [self.batch_size, 32, 32, 48], 2, 2, 2, 2, name='M_deconv2')
            net = tf.maximum(relu(conv2d(net, 96, 5, 5, 1, 1, name='M_conv2')),1e-6)

            net = tf.concat([net, en[0]], 3)
            net = deconv2d(net, [self.batch_size, 64, 64, 96], 2, 2, 2, 2, name='M_deconv3')
            net = tf.maximum(relu(conv2d(net, 96, 5, 5, 1, 1, name='M_conv3')),1e-6)

            net = hardtanh(conv2d(net, 1, 3, 3, 1, 1, name='M_conv4')) # 100 64 64 1
            # print net.get_shape()

            net = tf.concat([net, net, net], 3) # 100 64 64 3
            # print net.get_shape()

        return net

    # decoder for L
    def decoderL(self, latent, name, reuse=False):
        with tf.variable_scope(name):

            Lr = tf.contrib.layers.fully_connected(latent, 10, activation_fn=None)
            Lg = tf.contrib.layers.fully_connected(latent, 10, activation_fn=None)
            Lb = tf.contrib.layers.fully_connected(latent, 10, activation_fn=None)

            L = tf.stack([Lr, Lg, Lb], 2)

            # temp = L[:,0:9]
            # print temp.get_shape()
        return L

    # decoder for A
    def decoderA(self, latent, name, reuse=False):
        with tf.variable_scope(name):

            net = tf.maximum(tf.contrib.layers.fully_connected(latent, 24*8*8, activation_fn=None),1e-6)
            net = tf.reshape(net,[self.batch_size, 8, 8, 24])

            net = deconv2d(net, [self.batch_size, 16, 16, 24], 2, 2, 2, 2, name='A_deconv1')
            net = tf.maximum(relu(conv2d(net, 48, 5, 5, 1, 1, name='A_conv1')),1e-6)

            net = deconv2d(net, [self.batch_size, 32, 32, 48], 2, 2, 2, 2, name='A_deconv2')
            net = tf.maximum(relu(conv2d(net, 96, 5, 5, 1, 1, name='A_conv2')),1e-6)

            net = deconv2d(net, [self.batch_size, 64, 64, 96], 2, 2, 2, 2, name='A_deconv3')
            net = tf.maximum(relu(conv2d(net, 96, 5, 5, 1, 1, name='A_conv3')),1e-6)

            net = tf.maximum(conv2d(net, 3, 3, 3, 1, 1, name='A_conv4') ,1e-6)

            # replace maximum to hardtanh (too bright value)

        return net

    # decoder for N
    def decoderN(self, latent, name, reuse=False):
        with tf.variable_scope(name):

            net = tf.tanh(tf.contrib.layers.fully_connected(latent, 24*8*8, activation_fn=None))
            net = tf.reshape(net,[self.batch_size, 8, 8, 24])

            net = deconv2d(net, [self.batch_size, 16, 16, 24], 2, 2, 2, 2, name='N_deconv1')
            net = tf.tanh(relu(conv2d(net, 48, 5, 5, 1, 1, name='N_conv1')))

            net = deconv2d(net, [self.batch_size, 32, 32, 48], 2, 2, 2, 2, name='N_deconv2')
            net = tf.tanh(relu(conv2d(net, 96, 5, 5, 1, 1, name='N_conv2')))

            net = deconv2d(net, [self.batch_size, 64, 64, 96], 2, 2, 2, 2, name='N_deconv3')
            net = tf.tanh(relu(conv2d(net, 96, 5, 5, 1, 1, name='N_conv3')))

            net = conv2d(net, 2, 3, 3, 1, 1, name='N_conv4')

            N_at = net # 100*64*64*2

            Nxp, Nyp = tf.split(N_at, num_or_size_splits=2, axis=3) # 100 64 64 1

            Nxp = hardtanh(Nxp) # [-1 1]
            Nyp = hardtanh(Nyp) # [-1 1]

            Nxpsq = tf.square(Nxp) #nxpsq = nxp^2 > 0 always
            Nypsq = tf.square(Nyp)
            Nzpsq = tf.maximum(tf.nn.relu((Nxpsq+Nypsq-1)*(-1)),0) # 1 - nxp^2 - nyp^2 = nzp^2 > 0

            # Nzp = tf.maximum(tf.sqrt(Nzpsq),0)
            Nzp = hardtanh(tf.sqrt(Nzpsq)) # [-1 1]

            N = tf.concat([Nxp, Nyp, Nzp], axis=3) # batch*64*64*3

            Nnm = tf.expand_dims(tf.square(tf.norm(N, ord = 2, axis=3)),axis=3) # 100 64 64 1

            #normalize
            # Nn = hardtanh(normalizenormal(N)) # [-1 1] range, normalize to 1

        return N, Nnm


    def decoderO(self, bg, mask, light, normal, albedo, name, reuse=False):
        with tf.variable_scope(name):
            # print normal.get_shape() # 100 64 64 3
            # print light.get_shape() # 100 10 3

            # shading = SHPartialShadingRGB_bw2(normal, light) # 100 batch *64*64*3
            shading = SHPartialShadingRGB_bw2(normal, light) # 100 batch *64*64*3
            # print shading.get_shape() # 100 64 64 3

            # shading = relu(shading)
            shading = hardtanh(shading) #reasonable mean ground truth shading range in [0 1]

            # fg2 = albedo*(shading+1)/(2+1e-6)
            # fg2 = hardtanh(fg2)

            fg = albedo*shading
            fg = hardtanh(fg)

            output = fg*mask + bg*(1-mask)
            output = hardtanh(output)

        return shading, fg, output


    def discriminator(self, image, name, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            # if reuse:
                # scope.reuse_variables()

            # D-encoder
            d_enc = tf.maximum(conv2d(conv2d(image, 96, 5, 5, 1, 1, name='disc_conv1'), 96, 2, 2, 2, 2, name='disc_conv12'),1e-6)
            d_enc = tf.maximum(conv2d(conv2d(d_enc, 48, 5, 5, 1, 1, name='disc_conv2'), 48, 2, 2, 2, 2, name='disc_conv22'),1e-6)
            d_enc = tf.maximum(conv2d(conv2d(d_enc, 24, 5, 5, 1, 1, name='disc_conv3'), 24, 2, 2, 2, 2, name='disc_conv32'),1e-6)
            d_enc = tf.contrib.layers.fully_connected(tf.reshape(d_enc,[image.shape[0], -1]), 128, activation_fn=None)
            # d_enc = tf.contrib.layers.fully_connected(tf.reshape(d_enc,[image.shape[0], -1]), 128, activation_fn=tf.sigmoid)

            # D-code
            d_z = tf.sigmoid(d_enc) # [0 1]

            # D-decoder
            # d_dec = tf.maximum(tf.contrib.layers.fully_connected(d_enc, 24*8*8, activation_fn=tf.nn.relu),1e-6)
            d_dec = tf.maximum(tf.contrib.layers.fully_connected(d_z, 24*8*8, activation_fn=None),1e-6)
            d_dec = tf.reshape(d_dec,[self.batch_size, 8, 8, 24]) # 1 8 8 24
            d_dec = deconv2d(d_dec, [self.batch_size, 16, 16, 24], 2, 2, 2, 2, name='disc_dconv1') # maxunpool # 1 16 16 24
            d_dec = tf.maximum(relu(conv2d(d_dec, 48, 5, 5, 1, 1, name='disc_reconv1')),1e-6) # conv2d # 1 16 16 48
            d_dec = deconv2d(d_dec, [self.batch_size, 32, 32, 48], 2, 2, 2, 2, name='disc_dconv2') # 1 32 32 48
            d_dec = tf.maximum(relu(conv2d(d_dec, 96, 5, 5, 1, 1, name='disc_reconv2')),1e-6) # 1 32 32 96
            d_dec = deconv2d(d_dec, [self.batch_size, 64, 64, 96], 2, 2, 2, 2, name='disc_dconv3') # 1 64 64 96
            d_dec = tf.maximum(relu(conv2d(d_dec, 96, 5, 5, 1, 1, name='disc_reconv3')),1e-6) # 1 64 64 96

            # d_output = conv2d(d_dec, 3, 3, 3, 1, 1, name='disc_reconv4')# 1 64 64 3
            d_output = hardtanh(conv2d(d_dec, 3, 3, 3, 1, 1, name='disc_reconv4')) # 1 64 64 3
            # d_output = tf.maximum(conv2d(d_dec, 3, 3, 3, 1, 1, name='disc_reconv4'),1e-6) # 1 64 64 3

        return d_output
