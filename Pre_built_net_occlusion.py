import sys
import matplotlib.pyplot as plt
import numpy as np
import caffe

def ref_net_img_processing(net, caffe_root):
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    return transformer

def open_transform_img(transformer, i, caffe_root, input):
    number_zeros = 8 - len(str(i))
    img_number = number_zeros*'0' + str(i)
    # image = caffe.io.load_image(caffe_root + 'examples/images/ILSVRC2012_val_%s.jpeg' %(img_number))
    image = caffe.io.load_image('E:/Imagenet/ILSVRC2012_val_%s.jpeg' %(img_number))
    transformed_image = transformer.preprocess('data', image)
    input.append(transformed_image)
    i += 1
    return input, i

def net_forward(net, input):
    net.blobs['data'].data[...] = input
    out = net.forward()
    return out['prob']

def mix_images(input, percent_mix, filters):
    data_sets =[[],[],[],[]]
    # data_sets =[[]]
    for img in input:
        i = 0
        for filter in filters:
            # if i == 1:
            #     mix_percent = 1.0-percent_mix
            #     combined = (mix_percent*filter) + ((1-mix_percent)*img)
            #     data_sets[i].append(combined)
            #     i +=1
            # if i == 4:
            #     combined = filter + img
            #     data_sets[i].append(combined)
            #     i += 1

            combined = (percent_mix*filter) + ((1-percent_mix)*img)
            data_sets[i].append(combined)
            i +=1
    return data_sets[0], data_sets[1], data_sets[2], data_sets[3]
    # return data_sets[0]

def noisy(image):
    row,col,ch= image.shape
    mean = 0.5
    var = 0.05
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    g_noise = gauss.reshape(row,col,ch)
    #also generate totally random noise
    random = np.random.random_integers(0,1, (row,col,ch))
    return random, g_noise

def main():
    caffe_root = '../'
    sys.path.insert(0, caffe_root + 'python')
    caffe.set_mode_cpu()

    # print '##################################'
    # print 'Ref_net'
    #
    #
    # Ref_net = True
    # GoogleNet = False
    # ResNet = False
    #
    # if Ref_net:
    #     model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
    #     model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    # if GoogleNet:
    #     model_def = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
    #     model_weights = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
    # if ResNet:
    #     model_def = caffe_root + 'models/ResNet/ResNet-101-deploy.prototxt'
    #     model_weights = caffe_root + 'models/ResNet/ResNet-101-model.caffemodel'
    #
    # # Define the net, batch size and data input shape
    # net = caffe.Net(model_def,model_weights,caffe.TEST)
    #
    # BATCH_SIZE = 20 # how many images per batch
    # DATA_SIZE = 10000 # how many of the images will be processed
    # # Set up correct input batch size
    # if Ref_net:
    #     net.blobs['data'].reshape(BATCH_SIZE,3,227, 227)
    # if GoogleNet:
    #     net.blobs['data'].reshape(BATCH_SIZE,3,224, 224)
    # if ResNet:
    #     net.blobs['data'].reshape(BATCH_SIZE,3,224, 224)
    #
    # # instantiate the proper transformer for the net to preprocess the input images
    # transformer = ref_net_img_processing(net, caffe_root)
    #
    # # load the labels for the validation data
    # val_data = caffe_root + 'data/ilsvrc12/val.txt'
    # val_labels = np.loadtxt(val_data, str, delimiter='\t')
    #
    # #load labels
    # labels = np.loadtxt(caffe_root+"data/ilsvrc12/synset_words.txt", str, delimiter='\t')
    #
    # # different matricies to transform data
    # cat_image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
    # stapler_image = caffe.io.load_image(caffe_root + 'examples/images/stapler.jpg')
    # stapler_mixing_image = transformer.preprocess('data', stapler_image)
    # tribble_image = caffe.io.load_image(caffe_root + 'examples/images/tribble.jpg')
    # tribble_mixing_image = transformer.preprocess('data', tribble_image)
    # pedal_image = caffe.io.load_image(caffe_root + 'examples/images/PedalPD543.jpg')
    # pedal_mixing_image = transformer.preprocess('data', pedal_image)
    #
    # cat_mixing_image = transformer.preprocess('data', cat_image)
    # all_white = np.ones((cat_mixing_image.shape[1],cat_mixing_image.shape[1],cat_mixing_image.shape[0]))
    # white_trans = transformer.preprocess('data', all_white)
    # all_black = np.zeros((cat_mixing_image.shape[1],cat_mixing_image.shape[1],cat_mixing_image.shape[0]))
    # black_trans = transformer.preprocess('data', all_black)
    #
    # noise, gauss = noisy(cat_image)
    # trans_noise = transformer.preprocess('data', noise)
    # trans_gauss = transformer.preprocess('data', gauss)
    #
    # filters = [cat_mixing_image, white_trans, black_trans, trans_noise, trans_gauss]
    #
    # #####image mixing
    # percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # # unaltered = []
    # # cat = []
    # # white = []
    # # black = []
    # # random = []
    # # gaussian = []
    # tribble = []
    # stapler = []
    # pedal = []
    # for percent_mix in percents:
    #     #image counter
    #     i = 1
    #     top_one_1 = 0
    #     top_five_1 = 0
    #     total_images = 0
    #     top_one_2 = 0
    #     top_five_2 = 0
    #     top_one_3 = 0
    #     top_five_3 = 0
    #     # top_one_4 = 0
    #     # top_five_4 = 0
    #     # top_one_5 = 0
    #     # top_five_5 = 0
    #     # top_one_6 = 0
    #     # top_five_6 = 0
    #     print 'FOR MIXING PERCENT: ', percent_mix, '\n'
    #     ##### reading data, altering and predicting and getting accuracy
    #     input = []
    #     while i < DATA_SIZE+1:
    #         if i % BATCH_SIZE == 0:
    #             input, i = open_transform_img(transformer, i, caffe_root, input)
    #             out1 = net_forward(net,input) # give the batch to the net, forward prop and get outputs
    #             for ele in zip(out1, val_labels[i-BATCH_SIZE-1:i]):
    #                 total_images += 1
    #                 if total_images % 5000 == 0:
    #                     print total_images
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_1 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_1 += 1
    #             in_cat, in_white, in_black, in_random, in_gauss = mix_images(input,percent_mix, filters)
    #             out2 = net_forward(net,in_cat)
    #             for ele in zip(out2, val_labels[i-BATCH_SIZE-1:i]):
    #                 # print 'CAT Guess: ',ele[0].argmax(), 'CAT top five choices: ', ele[0].flatten().argsort()[-1:-6:-1] \
    #                 #         , 'CAT correct label: ',ele[1].split()[1]
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_2 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_2 += 1
    #             out3 = net_forward(net,in_white)
    #             for ele in zip(out3, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_3 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_3 += 1
    #             out4 = net_forward(net,in_black)
    #             # for ele in in_black:
    #             #     img = transformer.deprocess('data',ele)
    #             #     plt.imshow(img)
    #             #     plt.show()
    #             for ele in zip(out4, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_4 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_4 += 1
    #             out5 = net_forward(net,in_random)
    #             for ele in zip(out5, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_5 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_5 += 1
    #             out6 = net_forward(net,in_gauss)
    #             for ele in zip(out6, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_6 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_6 += 1
    #             input = []
    #         else:
    #             input, i = open_transform_img(transformer, i, caffe_root, input)
    #     print 'total wrong (no alter): ', top_one_1, 'not in top five (no alter): ', top_five_1, 'total images: ', total_images
    #     print 'top-one (no alter) score: ', float(top_one_1)/float(total_images)
    #     print 'top-five (no_alter) score: ', float(top_five_1)/float(total_images)
    #     unaltered.append((percent_mix,float(top_one_1)/float(total_images),float(top_five_1)/float(total_images)))
    #
    #     print 'total wrong (Cat): ', top_one_2, 'not in top five (cat): ', top_five_2, 'total images: ', total_images
    #     print 'top-one (Cat) score: ', float(top_one_2)/float(total_images)
    #     print 'top-five (Cat) score: ', float(top_five_2)/float(total_images)
    #     cat.append((percent_mix,float(top_one_2)/float(total_images),float(top_five_2)/float(total_images)))
    #
    #     print 'total wrong (white): ', top_one_3, 'not in top five (white): ', top_five_3, 'total images: ', total_images
    #     print 'top-one (white) score: ', float(top_one_3)/float(total_images)
    #     print 'top-five (white) score: ', float(top_five_3)/float(total_images)
    #     white.append((percent_mix,float(top_one_3)/float(total_images),float(top_five_3)/float(total_images)))
    #
    #     print 'total wrong (black): ', top_one_4, 'not in top five (black): ', top_five_4, 'total images: ', total_images
    #     print 'top-one (black) score: ', float(top_one_4)/float(total_images)
    #     print 'top-five (black) score: ', float(top_five_4)/float(total_images)
    #     black.append((percent_mix,float(top_one_4)/float(total_images),float(top_five_4)/float(total_images)))
    #
    #     print 'total wrong (random): ', top_one_5, 'not in top five (random): ', top_five_5, 'total images: ', total_images
    #     print 'top-one (random) score: ', float(top_one_5)/float(total_images)
    #     print 'top-five (random) score: ', float(top_five_5)/float(total_images)
    #     random.append((percent_mix,float(top_one_5)/float(total_images),float(top_five_5)/float(total_images)))
    #
    #     print 'total wrong (gaussian): ', top_one_6, 'not in top five (gaussian): ', top_five_6, 'total images: ', total_images
    #     print 'top-one (gaussian) score: ', float(top_one_6)/float(total_images)
    #     print 'top-five (gaussian) score: ', float(top_five_6)/float(total_images)
    #     gaussian.append((percent_mix,float(top_one_6)/float(total_images),float(top_five_6)/float(total_images)))
    #     # top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    #     # print labels[top_k]
    # print 'unaltered', unaltered
    # print 'cat', cat
    # print 'white', white
    # print 'black', black
    # print 'random', random
    # print 'gaussian', gaussian


    #
    # print '##################################'
    # print 'google net'
    #
    #
    # Ref_net = False
    # GoogleNet = True
    # ResNet = False
    #
    # if Ref_net:
    #     model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
    #     model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    # if GoogleNet:
    #     model_def = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
    #     model_weights = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
    # if ResNet:
    #     model_def = caffe_root + 'models/ResNet/ResNet-101-deploy.prototxt'
    #     model_weights = caffe_root + 'models/ResNet/ResNet-101-model.caffemodel'
    #
    # # Define the net, batch size and data input shape
    # net = caffe.Net(model_def,model_weights,caffe.TEST)
    #
    # BATCH_SIZE = 20 # how many images per batch
    # DATA_SIZE = 1000 # how many of the images will be processed
    # # Set up correct input batch size
    # if Ref_net:
    #     net.blobs['data'].reshape(BATCH_SIZE,3,227, 227)
    # if GoogleNet:
    #     net.blobs['data'].reshape(BATCH_SIZE,3,224, 224)
    # if ResNet:
    #     net.blobs['data'].reshape(BATCH_SIZE,3,224, 224)
    #
    # # instantiate the proper transformer for the net to preprocess the input images
    # transformer = ref_net_img_processing(net, caffe_root)
    #
    # # load the labels for the validation data
    # val_data = caffe_root + 'data/ilsvrc12/val.txt'
    # val_labels = np.loadtxt(val_data, str, delimiter='\t')
    #
    # #load labels
    # labels = np.loadtxt(caffe_root+"data/ilsvrc12/synset_words.txt", str, delimiter='\t')
    #
    # # different matricies to transform data
    # cat_image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
    # stapler_image = caffe.io.load_image(caffe_root + 'examples/images/stapler.jpg')
    # stapler_mixing_image = transformer.preprocess('data', stapler_image)
    # tribble_image = caffe.io.load_image(caffe_root + 'examples/images/tribble.jpg')
    # tribble_mixing_image = transformer.preprocess('data', tribble_image)
    # pedal_image = caffe.io.load_image(caffe_root + 'examples/images/PedalPD543.jpg')
    # pedal_mixing_image = transformer.preprocess('data', pedal_image)
    #
    # cat_mixing_image = transformer.preprocess('data', cat_image)
    # all_white = np.ones((cat_mixing_image.shape[1],cat_mixing_image.shape[1],cat_mixing_image.shape[0]))
    # white_trans = transformer.preprocess('data', all_white)
    # all_black = np.zeros((cat_mixing_image.shape[1],cat_mixing_image.shape[1],cat_mixing_image.shape[0]))
    # black_trans = transformer.preprocess('data', all_black)
    #
    # noise, gauss = noisy(cat_image)
    # trans_noise = transformer.preprocess('data', noise)
    # trans_gauss = transformer.preprocess('data', gauss)
    #
    # filters = [cat_mixing_image, white_trans, black_trans, trans_noise, trans_gauss]
    #
    # #####image mixing
    #
    # percents = [0.2, 0.3, 0.5, 0.7, 0.8]
    # unaltered = []
    # cat = []
    # white = []
    # black = []
    # random = []
    # gaussian = []
    # tribble = []
    # stapler = []
    # pedal = []
    # for percent_mix in percents:
    #     #image counter
    #     i = 1
    #     top_one_1 = 0
    #     top_five_1 = 0
    #     total_images = 0
    #     top_one_2 = 0
    #     top_five_2 = 0
    #     top_one_3 = 0
    #     top_five_3 = 0
    #     top_one_4 = 0
    #     top_five_4 = 0
    #     top_one_5 = 0
    #     top_five_5 = 0
    #     top_one_6 = 0
    #     top_five_6 = 0
    #     print 'FOR MIXING PERCENT: ', percent_mix, '\n'
    #     ##### reading data, altering and predicting and getting accuracy
    #     input = []
    #     while i < DATA_SIZE+1:
    #         if i % BATCH_SIZE == 0:
    #             input, i = open_transform_img(transformer, i, caffe_root, input)
    #             out1 = net_forward(net,input) # give the batch to the net, forward prop and get outputs
    #             for ele in zip(out1, val_labels[i-BATCH_SIZE-1:i]):
    #                 total_images += 1
    #                 if total_images % 5000 == 0:
    #                     print total_images
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_1 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_1 += 1
    #             in_cat, in_white, in_black, in_random, in_gauss = mix_images(input,percent_mix, filters)
    #             out2 = net_forward(net,in_cat)
    #             for ele in zip(out2, val_labels[i-BATCH_SIZE-1:i]):
    #                 # print 'CAT Guess: ',ele[0].argmax(), 'CAT top five choices: ', ele[0].flatten().argsort()[-1:-6:-1] \
    #                 #         , 'CAT correct label: ',ele[1].split()[1]
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_2 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_2 += 1
    #             out3 = net_forward(net,in_white)
    #             for ele in zip(out3, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_3 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_3 += 1
    #             out4 = net_forward(net,in_black)
    #             # for ele in in_black:
    #             #     img = transformer.deprocess('data',ele)
    #             #     plt.imshow(img)
    #             #     plt.show()
    #             for ele in zip(out4, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_4 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_4 += 1
    #             out5 = net_forward(net,in_random)
    #             for ele in zip(out5, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_5 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_5 += 1
    #             out6 = net_forward(net,in_gauss)
    #             for ele in zip(out6, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_6 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_6 += 1
    #             input = []
    #         else:
    #             input, i = open_transform_img(transformer, i, caffe_root, input)
    #     print 'total wrong (no alter): ', top_one_1, 'not in top five (no alter): ', top_five_1, 'total images: ', total_images
    #     print 'top-one (no alter) score: ', float(top_one_1)/float(total_images)
    #     print 'top-five (no_alter) score: ', float(top_five_1)/float(total_images)
    #     unaltered.append((percent_mix,float(top_one_1)/float(total_images),float(top_five_1)/float(total_images)))
    #
    #     print 'total wrong (Cat): ', top_one_2, 'not in top five (cat): ', top_five_2, 'total images: ', total_images
    #     print 'top-one (Cat) score: ', float(top_one_2)/float(total_images)
    #     print 'top-five (Cat) score: ', float(top_five_2)/float(total_images)
    #     cat.append((percent_mix,float(top_one_2)/float(total_images),float(top_five_2)/float(total_images)))
    #
    #     print 'total wrong (white): ', top_one_3, 'not in top five (white): ', top_five_3, 'total images: ', total_images
    #     print 'top-one (white) score: ', float(top_one_3)/float(total_images)
    #     print 'top-five (white) score: ', float(top_five_3)/float(total_images)
    #     white.append((percent_mix,float(top_one_3)/float(total_images),float(top_five_3)/float(total_images)))
    #
    #     print 'total wrong (black): ', top_one_4, 'not in top five (black): ', top_five_4, 'total images: ', total_images
    #     print 'top-one (black) score: ', float(top_one_4)/float(total_images)
    #     print 'top-five (black) score: ', float(top_five_4)/float(total_images)
    #     black.append((percent_mix,float(top_one_4)/float(total_images),float(top_five_4)/float(total_images)))
    #
    #     print 'total wrong (random): ', top_one_5, 'not in top five (random): ', top_five_5, 'total images: ', total_images
    #     print 'top-one (random) score: ', float(top_one_5)/float(total_images)
    #     print 'top-five (random) score: ', float(top_five_5)/float(total_images)
    #     random.append((percent_mix,float(top_one_5)/float(total_images),float(top_five_5)/float(total_images)))
    #
    #     print 'total wrong (gaussian): ', top_one_6, 'not in top five (gaussian): ', top_five_6, 'total images: ', total_images
    #     print 'top-one (gaussian) score: ', float(top_one_6)/float(total_images)
    #     print 'top-five (gaussian) score: ', float(top_five_6)/float(total_images)
    #     gaussian.append((percent_mix,float(top_one_6)/float(total_images),float(top_five_6)/float(total_images)))
    #     # top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    #     # print labels[top_k]
    # print 'unaltered', unaltered
    # print 'cat', cat
    # print 'white', white
    # print 'black', black
    # print 'random', random
    # print 'gaussian', gaussian









    # print '##################################'
    # print 'GoogleNet and resnet'
    #
    #
    # Ref_net = False
    # GoogleNet = True
    # ResNet = True
    #
    # if Ref_net:
    #     model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
    #     model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    # if GoogleNet:
    #     model_def = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
    #     model_weights = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
    # if ResNet:
    #     model_def2 = caffe_root + 'models/ResNet/ResNet-50-deploy.prototxt'
    #     model_weights2 = caffe_root + 'models/ResNet/ResNet-50-model.caffemodel'
    #
    # # Define the net, batch size and data input shape
    # net = caffe.Net(model_def,model_weights,caffe.TEST)
    # net2 = caffe.Net(model_def2,model_weights2,caffe.TEST)
    #
    # BATCH_SIZE = 20 # how many images per batch
    # DATA_SIZE = 500 # how many of the images will be processed
    # # Set up correct input batch size
    # if Ref_net:
    #     net.blobs['data'].reshape(BATCH_SIZE,3,227, 227)
    # if GoogleNet:
    #     net.blobs['data'].reshape(BATCH_SIZE,3,224, 224)
    # if ResNet:
    #     net2.blobs['data'].reshape(BATCH_SIZE,3,224, 224)
    #
    # # instantiate the proper transformer for the net to preprocess the input images
    # transformer = ref_net_img_processing(net, caffe_root)
    # transformer2 = ref_net_img_processing(net2, caffe_root)
    # # load the labels for the validation data
    # val_data = caffe_root + 'data/ilsvrc12/val.txt'
    # val_labels = np.loadtxt(val_data, str, delimiter='\t')
    #
    # #load labels
    # labels = np.loadtxt(caffe_root+"data/ilsvrc12/synset_words.txt", str, delimiter='\t')
    #
    # # different matricies to transform data
    # cat_image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
    # cat_mixing_image = transformer.preprocess('data', cat_image)
    # all_white = np.ones((cat_mixing_image.shape[1],cat_mixing_image.shape[1],cat_mixing_image.shape[0]))
    # white_trans = transformer.preprocess('data', all_white)
    # all_black = np.zeros((cat_mixing_image.shape[1],cat_mixing_image.shape[1],cat_mixing_image.shape[0]))
    # black_trans = transformer.preprocess('data', all_black)
    #
    # noise, gauss = noisy(cat_image)
    # trans_noise = transformer.preprocess('data', noise)
    # trans_gauss = transformer.preprocess('data', gauss)
    #
    # filters = [cat_mixing_image, white_trans, black_trans, trans_noise, trans_gauss]
    #
    # #####image mixing
    # percents = [0.2, 0.3, 0.5, 0.7, 0.8]
    # unaltered = []
    # cat = []
    # white = []
    # black = []
    # random = []
    # gaussian = []
    # unaltered2 = []
    # cat2 = []
    # white2 = []
    # black2 = []
    # random2 = []
    # gaussian2 = []
    # for percent_mix in percents:
    #     #image counter
    #     i = 1
    #     top_one_1 = 0
    #     top_five_1 = 0
    #     total_images = 0
    #     top_one_2 = 0
    #     top_five_2 = 0
    #     top_one_3 = 0
    #     top_five_3 = 0
    #     top_one_4 = 0
    #     top_five_4 = 0
    #     top_one_5 = 0
    #     top_five_5 = 0
    #     top_one_6 = 0
    #     top_five_6 = 0
    #     top_one_1_2 = 0
    #     top_five_1_2 = 0
    #     top_one_2_2 = 0
    #     top_five_2_2 = 0
    #     top_one_3_2 = 0
    #     top_five_3_2 = 0
    #     top_one_4_2 = 0
    #     top_five_4_2 = 0
    #     top_one_5_2 = 0
    #     top_five_5_2 = 0
    #     top_one_6_2 = 0
    #     top_five_6_2 = 0
    #     print 'FOR MIXING PERCENT: ', percent_mix, '\n'
    #     ##### reading data, altering and predicting and getting accuracy
    #     input = []
    #     while i < DATA_SIZE+1:
    #         if i % BATCH_SIZE == 0:
    #             input, i = open_transform_img(transformer, i, caffe_root, input)
    #             out1 = net_forward(net,input) # give the batch to the net, forward prop and get outputs
    #             out1_2 = net_forward(net2,input)
    #             for ele in zip(out1, val_labels[i-BATCH_SIZE-1:i]):
    #                 total_images += 1
    #                 if total_images % 20 == 0:
    #                     print total_images
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_1 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_1 += 1
    #             for ele in zip(out1_2, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_1_2 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_1_2 += 1
    #             in_cat, in_white, in_black, in_random, in_gauss = mix_images(input,percent_mix, filters)
    #             out2 = net_forward(net,in_cat)
    #             out2_2 = net_forward(net2,in_cat)
    #             for ele in zip(out2, val_labels[i-BATCH_SIZE-1:i]):
    #                 # print 'CAT Guess: ',ele[0].argmax(), 'CAT top five choices: ', ele[0].flatten().argsort()[-1:-6:-1] \
    #                 #         , 'CAT correct label: ',ele[1].split()[1]
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_2 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_2 += 1
    #             for ele in zip(out2_2, val_labels[i-BATCH_SIZE-1:i]):
    #                 # print 'CAT Guess: ',ele[0].argmax(), 'CAT top five choices: ', ele[0].flatten().argsort()[-1:-6:-1] \
    #                 #         , 'CAT correct label: ',ele[1].split()[1]
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_2_2 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_2_2 += 1
    #             out3 = net_forward(net,in_white)
    #             out3_2 = net_forward(net2,in_white)
    #             for ele in zip(out3, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_3 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_3 += 1
    #             for ele in zip(out3_2, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_3_2 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_3_2 += 1
    #             out4 = net_forward(net,in_black)
    #             out4_2 = net_forward(net2,in_black)
    #             # for ele in in_black:
    #             #     img = transformer.deprocess('data',ele)
    #             #     plt.imshow(img)
    #             #     plt.show()
    #             for ele in zip(out4, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_4 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_4 += 1
    #             for ele in zip(out4_2, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_4_2 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_4_2 += 1
    #             out5 = net_forward(net,in_random)
    #             out5_2 = net_forward(net2,in_random)
    #             for ele in zip(out5, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_5 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_5 += 1
    #             for ele in zip(out5_2, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_5_2 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_5_2 += 1
    #             out6 = net_forward(net,in_gauss)
    #             out6_2 = net_forward(net2,in_gauss)
    #             for ele in zip(out6, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_6 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_6 += 1
    #             for ele in zip(out6_2, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_6_2 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_6_2 += 1
    #             input = []
    #         else:
    #             input, i = open_transform_img(transformer, i, caffe_root, input)
    #     print 'total wrong (no alter): ', top_one_1, 'not in top five (no alter): ', top_five_1, 'total images: ', total_images
    #     print 'top-one (no alter) score: ', float(top_one_1)/float(total_images)
    #     print 'top-five (no_alter) score: ', float(top_five_1)/float(total_images)
    #     unaltered.append((percent_mix,float(top_one_1)/float(total_images),float(top_five_1)/float(total_images)))
    #
    #     print 'total wrong (Cat): ', top_one_2, 'not in top five (cat): ', top_five_2, 'total images: ', total_images
    #     print 'top-one (Cat) score: ', float(top_one_2)/float(total_images)
    #     print 'top-five (Cat) score: ', float(top_five_2)/float(total_images)
    #     cat.append((percent_mix,float(top_one_2)/float(total_images),float(top_five_2)/float(total_images)))
    #
    #     print 'total wrong (white): ', top_one_3, 'not in top five (white): ', top_five_3, 'total images: ', total_images
    #     print 'top-one (white) score: ', float(top_one_3)/float(total_images)
    #     print 'top-five (white) score: ', float(top_five_3)/float(total_images)
    #     white.append((percent_mix,float(top_one_3)/float(total_images),float(top_five_3)/float(total_images)))
    #
    #     print 'total wrong (black): ', top_one_4, 'not in top five (black): ', top_five_4, 'total images: ', total_images
    #     print 'top-one (black) score: ', float(top_one_4)/float(total_images)
    #     print 'top-five (black) score: ', float(top_five_4)/float(total_images)
    #     black.append((percent_mix,float(top_one_4)/float(total_images),float(top_five_4)/float(total_images)))
    #
    #     print 'total wrong (random): ', top_one_5, 'not in top five (random): ', top_five_5, 'total images: ', total_images
    #     print 'top-one (random) score: ', float(top_one_5)/float(total_images)
    #     print 'top-five (random) score: ', float(top_five_5)/float(total_images)
    #     random.append((percent_mix,float(top_one_5)/float(total_images),float(top_five_5)/float(total_images)))
    #
    #     print 'total wrong (gaussian): ', top_one_6, 'not in top five (gaussian): ', top_five_6, 'total images: ', total_images
    #     print 'top-one (gaussian) score: ', float(top_one_6)/float(total_images)
    #     print 'top-five (gaussian) score: ', float(top_five_6)/float(total_images)
    #     gaussian.append((percent_mix,float(top_one_6)/float(total_images),float(top_five_6)/float(total_images)))
    #     # top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    #     # print labels[top_k]
    #
    #     print '############RES NET'
    #     print 'total wrong (no alter): ', top_one_1_2, 'not in top five (no alter): ', top_five_1_2, 'total images: ', total_images
    #     print 'top-one (no alter) score: ', float(top_one_1_2)/float(total_images)
    #     print 'top-five (no_alter) score: ', float(top_five_1_2)/float(total_images)
    #     unaltered2.append((percent_mix,float(top_one_1_2)/float(total_images),float(top_five_1_2)/float(total_images)))
    #
    #     print 'total wrong (Cat): ', top_one_2_2, 'not in top five (cat): ', top_five_2_2, 'total images: ', total_images
    #     print 'top-one (Cat) score: ', float(top_one_2_2)/float(total_images)
    #     print 'top-five (Cat) score: ', float(top_five_2_2)/float(total_images)
    #     cat2.append((percent_mix,float(top_one_2_2)/float(total_images),float(top_five_2_2)/float(total_images)))
    #
    #     print 'total wrong (white): ', top_one_3_2, 'not in top five (white): ', top_five_3_2, 'total images: ', total_images
    #     print 'top-one (white) score: ', float(top_one_3_2)/float(total_images)
    #     print 'top-five (white) score: ', float(top_five_3_2)/float(total_images)
    #     white2.append((percent_mix,float(top_one_3_2)/float(total_images),float(top_five_3_2)/float(total_images)))
    #
    #     print 'total wrong (black): ', top_one_4_2, 'not in top five (black): ', top_five_4_2, 'total images: ', total_images
    #     print 'top-one (black) score: ', float(top_one_4_2)/float(total_images)
    #     print 'top-five (black) score: ', float(top_five_4_2)/float(total_images)
    #     black2.append((percent_mix,float(top_one_4_2)/float(total_images),float(top_five_4_2)/float(total_images)))
    #
    #     print 'total wrong (random): ', top_one_5_2, 'not in top five (random): ', top_five_5_2, 'total images: ', total_images
    #     print 'top-one (random) score: ', float(top_one_5_2)/float(total_images)
    #     print 'top-five (random) score: ', float(top_five_5_2)/float(total_images)
    #     random2.append((percent_mix,float(top_one_5_2)/float(total_images),float(top_five_5_2)/float(total_images)))
    #
    #     print 'total wrong (gaussian): ', top_one_6_2, 'not in top five (gaussian): ', top_five_6_2, 'total images: ', total_images
    #     print 'top-one (gaussian) score: ', float(top_one_6_2)/float(total_images)
    #     print 'top-five (gaussian) score: ', float(top_five_6_2)/float(total_images)
    #     gaussian2.append((percent_mix,float(top_one_6_2)/float(total_images),float(top_five_6_2)/float(total_images)))
    # print 'unaltered', unaltered2
    # print 'cat', cat2
    # print 'white', white2
    # print 'black', black2
    # print 'random', random2
    # print 'gaussian', gaussian2













    print '##################################'
    print 'Resnet 50'

    Ref_net = False
    GoogleNet = False
    ResNet = True

    if Ref_net:
        model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
        model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    if GoogleNet:
        model_def = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
        model_weights = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
    if ResNet:
        model_def = caffe_root + 'models/ResNet/ResNet-50-deploy.prototxt'
        model_weights = caffe_root + 'models/ResNet/ResNet-50-model.caffemodel'

    # Define the net, batch size and data input shape
    net = caffe.Net(model_def,model_weights,caffe.TEST)

    BATCH_SIZE = 5 # how many images per batch
    DATA_SIZE = 200 # how many of the images will be processed
    # Set up correct input batch size
    if Ref_net:
        net.blobs['data'].reshape(BATCH_SIZE,3,227, 227)
    if GoogleNet:
        net.blobs['data'].reshape(BATCH_SIZE,3,224, 224)
    if ResNet:
        net.blobs['data'].reshape(BATCH_SIZE,3,224, 224)

    # instantiate the proper transformer for the net to preprocess the input images
    transformer = ref_net_img_processing(net, caffe_root)

    # load the labels for the validation data
    val_data = caffe_root + 'data/ilsvrc12/val.txt'
    val_labels = np.loadtxt(val_data, str, delimiter='\t')

    #load labels
    labels = np.loadtxt(caffe_root+"data/ilsvrc12/synset_words.txt", str, delimiter='\t')

    # different matricies to transform data
    cat_image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
    cat_mixing_image = transformer.preprocess('data', cat_image)
    all_white = np.ones((cat_mixing_image.shape[1],cat_mixing_image.shape[1],cat_mixing_image.shape[0]))
    white_trans = transformer.preprocess('data', all_white)
    all_black = np.zeros((cat_mixing_image.shape[1],cat_mixing_image.shape[1],cat_mixing_image.shape[0]))
    black_trans = transformer.preprocess('data', all_black)

    noise, gauss = noisy(cat_image)
    trans_noise = transformer.preprocess('data', noise)
    trans_gauss = transformer.preprocess('data', gauss)

    filters = [cat_mixing_image, white_trans, trans_noise, trans_gauss]
    # filters = [black_trans]

    #####image mixing
    percents = [0.1, 0.2, 0.5, 0.7, 0.8]
    unaltered = []
    cat = []
    white = []
    black = []
    random = []
    gaussian = []
    for percent_mix in percents:
        #image counter
        i = 1
        top_one_1 = 0
        top_five_1 = 0
        total_images = 0
        top_one_2 = 0
        top_five_2 = 0
        top_one_3 = 0
        top_five_3 = 0
        top_one_4 = 0
        top_five_4 = 0
        top_one_5 = 0
        top_five_5 = 0
        top_one_6 = 0
        top_five_6 = 0
        print 'FOR MIXING PERCENT: ', percent_mix, '\n'
        ##### reading data, altering and predicting and getting accuracy
        input = []
        while i < DATA_SIZE+1:
            if i % BATCH_SIZE == 0:
                input, i = open_transform_img(transformer, i, caffe_root, input)
                # out1 = net_forward(net,input) # give the batch to the net, forward prop and get outputs
                # for ele in zip(out1, val_labels[i-BATCH_SIZE-1:i]):
                #     # if total_images % 20 == 0:
                #     #     print total_images
                #     if int(ele[0].argmax()) != int(ele[1].split()[1]):
                #         top_one_1 += 1
                #     if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
                #         top_five_1 += 1
                in_cat, in_white, in_random, in_gauss = mix_images(input,percent_mix, filters)
                out2 = net_forward(net,in_cat)
                for ele in zip(out2, val_labels[i-BATCH_SIZE-1:i]):
                    total_images += 1
                    # print 'CAT Guess: ',ele[0].argmax(), 'CAT top five choices: ', ele[0].flatten().argsort()[-1:-6:-1] \
                    #         , 'CAT correct label: ',ele[1].split()[1]
                    if int(ele[0].argmax()) != int(ele[1].split()[1]):
                        top_one_2 += 1
                    if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
                        top_five_2 += 1
                out3 = net_forward(net,in_white)
                for ele in zip(out3, val_labels[i-BATCH_SIZE-1:i]):
                    if int(ele[0].argmax()) != int(ele[1].split()[1]):
                        top_one_3 += 1
                    if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
                        top_five_3 += 1
                # out4 = net_forward(net,in_black)
                # # for ele in in_black:
                # #     img = transformer.deprocess('data',ele)
                # #     plt.imshow(img)
                # #     plt.show()
                # for ele in zip(out4, val_labels[i-BATCH_SIZE-1:i]):
                #     if int(ele[0].argmax()) != int(ele[1].split()[1]):
                #         top_one_4 += 1
                #     if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
                #         top_five_4 += 1
                out5 = net_forward(net,in_random)
                for ele in zip(out5, val_labels[i-BATCH_SIZE-1:i]):
                    if int(ele[0].argmax()) != int(ele[1].split()[1]):
                        top_one_5 += 1
                    if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
                        top_five_5 += 1
                out6 = net_forward(net,in_gauss)
                for ele in zip(out6, val_labels[i-BATCH_SIZE-1:i]):
                    if int(ele[0].argmax()) != int(ele[1].split()[1]):
                        top_one_6 += 1
                    if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
                        top_five_6 += 1
                input = []
            else:
                input, i = open_transform_img(transformer, i, caffe_root, input)
        # print 'total wrong (no alter): ', top_one_1, 'not in top five (no alter): ', top_five_1, 'total images: ', total_images
        # print 'top-one (no alter) score: ', float(top_one_1)/float(total_images)
        # print 'top-five (no_alter) score: ', float(top_five_1)/float(total_images)
        # unaltered.append((percent_mix,float(top_one_1)/float(total_images),float(top_five_1)/float(total_images)))

        print 'total wrong (black): ', top_one_2, 'not in top five (black): ', top_five_2, 'total images: ', total_images
        print 'top-one (black) score: ', float(top_one_2)/float(total_images)
        print 'top-five (black) score: ', float(top_five_2)/float(total_images)
        cat.append((percent_mix,float(top_one_2)/float(total_images),float(top_five_2)/float(total_images)))

        print 'total wrong (white): ', top_one_3, 'not in top five (white): ', top_five_3, 'total images: ', total_images
        print 'top-one (white) score: ', float(top_one_3)/float(total_images)
        print 'top-five (white) score: ', float(top_five_3)/float(total_images)
        white.append((percent_mix,float(top_one_3)/float(total_images),float(top_five_3)/float(total_images)))

        # print 'total wrong (black): ', top_one_4, 'not in top five (black): ', top_five_4, 'total images: ', total_images
        # print 'top-one (black) score: ', float(top_one_4)/float(total_images)
        # print 'top-five (black) score: ', float(top_five_4)/float(total_images)
        # black.append((percent_mix,float(top_one_4)/float(total_images),float(top_five_4)/float(total_images)))

        print 'total wrong (random): ', top_one_5, 'not in top five (random): ', top_five_5, 'total images: ', total_images
        print 'top-one (random) score: ', float(top_one_5)/float(total_images)
        print 'top-five (random) score: ', float(top_five_5)/float(total_images)
        random.append((percent_mix,float(top_one_5)/float(total_images),float(top_five_5)/float(total_images)))

        print 'total wrong (gaussian): ', top_one_6, 'not in top five (gaussian): ', top_five_6, 'total images: ', total_images
        print 'top-one (gaussian) score: ', float(top_one_6)/float(total_images)
        print 'top-five (gaussian) score: ', float(top_five_6)/float(total_images)
        gaussian.append((percent_mix,float(top_one_6)/float(total_images),float(top_five_6)/float(total_images)))
        # top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
        # print labels[top_k]
    # print 'unaltered', unaltered
    # print 'black', black
    # # print 'white', white
    # # print 'black', black
    # # print 'random', random
    # # print 'gaussian', gaussian


if __name__ == '__main__':
    main()