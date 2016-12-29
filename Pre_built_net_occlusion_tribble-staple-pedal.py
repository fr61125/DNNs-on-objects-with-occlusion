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
    # data_sets =[[],[],[],[],[]]
    # data_sets =[[],[],[]]
    data_sets = [[]]
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
    # return data_sets[0], data_sets[1], data_sets[2],  data_sets[3],  data_sets[4]
    # return data_sets[0], data_sets[1], data_sets[2]
    return data_sets[0]
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
    #
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
    #     model_def = caffe_root + 'models/ResNet/ResNet-50-deploy.prototxt'
    #     model_weights = caffe_root + 'models/ResNet/ResNet-50-model.caffemodel'
    #
    # # Define the net, batch size and data input shape
    # net = caffe.Net(model_def,model_weights,caffe.TEST)
    #
    # BATCH_SIZE = 10 # how many images per batch
    # DATA_SIZE = 5000 # how many of the images will be processed
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
    # cat_mixing_image = transformer.preprocess('data', cat_image)
    # stapler_image = caffe.io.load_image(caffe_root + 'examples/images/stapler.jpg')
    # stapler_mixing_image = transformer.preprocess('data', stapler_image)
    # tribble_image = caffe.io.load_image(caffe_root + 'examples/images/tribble.jpg')
    # tribble_mixing_image = transformer.preprocess('data', tribble_image)
    # pedal_image = caffe.io.load_image(caffe_root + 'examples/images/PedalPD543.jpg')
    # pedal_mixing_image = transformer.preprocess('data', pedal_image)
    #
    #
    #
    # filters = [tribble_mixing_image, stapler_mixing_image, pedal_mixing_image]

    # #####image mixing
    # percents = [0.1, 0.3, 0.5, 0.8]
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
    #     print 'FOR MIXING PERCENT: ', percent_mix, '\n'
    #     ##### reading data, altering and predicting and getting accuracy
    #     input = []
    #     while i < DATA_SIZE+1:
    #         if i % BATCH_SIZE == 0:
    #             input, i = open_transform_img(transformer, i, caffe_root, input)
    #             in_tribble, in_stapler, in_pedal = mix_images(input,percent_mix, filters)
    #             out1 = net_forward(net,in_tribble) # give the batch to the net, forward prop and get outputs
    #             for ele in zip(out1, val_labels[i-BATCH_SIZE-1:i]):
    #                 total_images += 1
    #                 if total_images % 5000 == 0:
    #                     print total_images
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_1 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_1 += 1
    #             out2 = net_forward(net,in_stapler)
    #             for ele in zip(out2, val_labels[i-BATCH_SIZE-1:i]):
    #                 # print 'CAT Guess: ',ele[0].argmax(), 'CAT top five choices: ', ele[0].flatten().argsort()[-1:-6:-1] \
    #                 #         , 'CAT correct label: ',ele[1].split()[1]
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_2 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_2 += 1
    #             out3 = net_forward(net,in_pedal)
    #             for ele in zip(out3, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_3 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_3 += 1
    #             input = []
    #         else:
    #             input, i = open_transform_img(transformer, i, caffe_root, input)
    #     print 'total wrong (tribble): ', top_one_1, 'not in top five (tribble): ', top_five_1, 'total images: ', total_images
    #     print 'top-one (tribble) score: ', float(top_one_1)/float(total_images)
    #     print 'top-five (tribble) score: ', float(top_five_1)/float(total_images)
    #     tribble.append((percent_mix,float(top_one_1)/float(total_images),float(top_five_1)/float(total_images)))
    #
    #     print 'total wrong (stapler): ', top_one_2, 'not in top five (stapler): ', top_five_2, 'total images: ', total_images
    #     print 'top-one (stapler) score: ', float(top_one_2)/float(total_images)
    #     print 'top-five (stapler) score: ', float(top_five_2)/float(total_images)
    #     stapler.append((percent_mix,float(top_one_2)/float(total_images),float(top_five_2)/float(total_images)))
    #
    #     print 'total wrong (pedal): ', top_one_3, 'not in top five (pedal): ', top_five_3, 'total images: ', total_images
    #     print 'top-one (pedal) score: ', float(top_one_3)/float(total_images)
    #     print 'top-five (pedal) score: ', float(top_five_3)/float(total_images)
    #     pedal.append((percent_mix,float(top_one_3)/float(total_images),float(top_five_3)/float(total_images)))
    #
    #
    #     # top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    #     # print labels[top_k]
    # print 'unaltered', tribble
    # print 'stapler', stapler
    # print 'pedal', pedal
    #
    #
    #
    print '##################################'
    print 'ResNet'

    Ref_net = False
    GoogleNet =True
    ResNet = False

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

    BATCH_SIZE = 2 # how many images per batch
    DATA_SIZE = 1000 # how many of the images will be processed
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
    stapler_image = caffe.io.load_image(caffe_root + 'examples/images/stapler.jpg')
    stapler_mixing_image = transformer.preprocess('data', stapler_image)
    tribble_image = caffe.io.load_image(caffe_root + 'examples/images/tribble.jpg')
    tribble_mixing_image = transformer.preprocess('data', tribble_image)
    pedal_image = caffe.io.load_image(caffe_root + 'examples/images/PedalPD543.jpg')
    pedal_mixing_image = transformer.preprocess('data', pedal_image)

    cat_mixing_image = transformer.preprocess('data', cat_image)
    all_white = np.ones((cat_mixing_image.shape[1],cat_mixing_image.shape[1],cat_mixing_image.shape[0]))
    white_trans = transformer.preprocess('data', all_white)
    all_black = np.zeros((cat_mixing_image.shape[1],cat_mixing_image.shape[1],cat_mixing_image.shape[0]))
    black_trans = transformer.preprocess('data', all_black)

    noise, gauss = noisy(cat_image)
    trans_noise = transformer.preprocess('data', noise)
    trans_gauss = transformer.preprocess('data', gauss)

    # filters = [cat_mixing_image, white_trans, black_trans, trans_noise, trans_gauss]
    # filters = [tribble_mixing_image, stapler_mixing_image, pedal_mixing_image]
    filters = [cat_mixing_image]

    #####image mixing
    percents = [0.1]
    # unaltered = []
    cat = []
    # white = []
    # black = []
    # random = []
    # gaussian = []
    # tribble = []
    # stapler = []
    # pedal = []
    for percent_mix in percents:
        #image counter
        i = 1
        top_one_1 = 0
        top_five_1 = 0
        total_images = 0
        # top_one_2 = 0
        # top_five_2 = 0
        # top_one_3 = 0
        # top_five_3 = 0
        # top_one_4 = 0
        # top_five_4 = 0
        # top_one_5 = 0
        # top_five_5 = 0
        # top_one_6 = 0
        # top_five_6 = 0
        print 'FOR MIXING PERCENT: ', percent_mix, '\n'
        ##### reading data, altering and predicting and getting accuracy
        input = []
        while i < DATA_SIZE+1:
            if i % BATCH_SIZE == 0:
                input, i = open_transform_img(transformer, i, caffe_root, input)
                in_cat = mix_images(input,percent_mix, filters)
                out1 = net_forward(net,in_cat) # give the batch to the net, forward prop and get outputs
                for ele in zip(out1, val_labels[i-BATCH_SIZE-1:i]):
                    total_images += 1
                    if total_images % 50 == 0:
                        print total_images
                    if int(ele[0].argmax()) != int(ele[1].split()[1]):
                        top_one_1 += 1
                    if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
                        top_five_1 += 1
                # out2 = net_forward(net,in_stapler)
                # for ele in zip(out2, val_labels[i-BATCH_SIZE-1:i]):
                #     # print 'CAT Guess: ',ele[0].argmax(), 'CAT top five choices: ', ele[0].flatten().argsort()[-1:-6:-1] \
                #     #         , 'CAT correct label: ',ele[1].split()[1]
                #     if int(ele[0].argmax()) != int(ele[1].split()[1]):
                #         top_one_2 += 1
                #     if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
                #         top_five_2 += 1
                # out3 = net_forward(net,in_pedal)
                # for ele in zip(out3, val_labels[i-BATCH_SIZE-1:i]):
                #     if int(ele[0].argmax()) != int(ele[1].split()[1]):
                #         top_one_3 += 1
                #     if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
                #         top_five_3 += 1
                input = []
            else:
                input, i = open_transform_img(transformer, i, caffe_root, input)
        print 'total wrong (cat): ', top_one_1, 'not in top five (tribble): ', top_five_1, 'total images: ', total_images
        print 'top-one (cat) score: ', float(top_one_1)/float(total_images)
        print 'top-five (cat) score: ', float(top_five_1)/float(total_images)
        cat.append((percent_mix,float(top_one_1)/float(total_images),float(top_five_1)/float(total_images)))

        # print 'total wrong (stapler): ', top_one_2, 'not in top five (stapler): ', top_five_2, 'total images: ', total_images
        # print 'top-one (stapler) score: ', float(top_one_2)/float(total_images)
        # print 'top-five (stapler) score: ', float(top_five_2)/float(total_images)
        # stapler.append((percent_mix,float(top_one_2)/float(total_images),float(top_five_2)/float(total_images)))
        #
        # print 'total wrong (pedal): ', top_one_3, 'not in top five (pedal): ', top_five_3, 'total images: ', total_images
        # print 'top-one (pedal) score: ', float(top_one_3)/float(total_images)
        # print 'top-five (pedal) score: ', float(top_five_3)/float(total_images)
        # pedal.append((percent_mix,float(top_one_3)/float(total_images),float(top_five_3)/float(total_images)))


        # top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
        # print labels[top_k]
    # print 'unaltered', tribble
    # print 'stapler', stapler
    # print 'pedal', pedal









    #
    #
    #
    # print '##################################'
    # print 'Resnet 50'
    #
    # Ref_net = False
    # GoogleNet = False
    # ResNet = True
    #
    # if Ref_net:
    #     model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
    #     model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    # if GoogleNet:
    #     model_def = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
    #     model_weights = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
    # if ResNet:
    #     model_def = caffe_root + 'models/ResNet/ResNet-50-deploy.prototxt'
    #     model_weights = caffe_root + 'models/ResNet/ResNet-50-model.caffemodel'
    #
    # # Define the net, batch size and data input shape
    # net = caffe.Net(model_def,model_weights,caffe.TEST)
    #
    # BATCH_SIZE = 1 # how many images per batch
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
    #
    # # strawberry_image = caffe.io.load_image(caffe_root + 'examples/images/ILSVRC2012_val_00000099.jpeg')
    # # strawberry_mixing_image = transformer.preprocess('data', strawberry_image)
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
    # # filters = [tribble_mixing_image, stapler_mixing_image, pedal_mixing_image]
    #
    # #####image mixing
    # percents = [0.3, 0.5, 0.7]
    # for percent_mix in percents:
    #     print percent_mix
    #     input = [strawberry_mixing_image]
    #     strawberry = transformer.deprocess('data', strawberry_mixing_image)
    #     plt.imshow(strawberry)
    #     plt.show()
    #     in_cat, in_white, in_black, in_random, in_gaussian = mix_images(input,percent_mix, filters)
    #     strawberry_cat = transformer.deprocess('data', in_cat[0])
    #     plt.imshow(strawberry_cat)
    #     plt.show()
    #     strawberry_black = transformer.deprocess('data', in_black[0])
    #     plt.imshow(strawberry_black)
    #     plt.show()
    #     strawberry_white = transformer.deprocess('data', in_white[0])
    #     plt.imshow(strawberry_white)
    #     plt.show()
    #     strawberry_random = transformer.deprocess('data', in_random[0])
    #     plt.imshow(strawberry_random)
    #     plt.show()
    #     strawberry_gaussian = transformer.deprocess('data', in_gaussian[0])
    #     plt.imshow(strawberry_gaussian)
    #     plt.show()
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
    #             in_tribble, in_stapler, in_pedal = mix_images(input,percent_mix, filters)
    #             out1 = net_forward(net,in_tribble) # give the batch to the net, forward prop and get outputs
    #             for ele in zip(out1, val_labels[i-BATCH_SIZE-1:i]):
    #                 total_images += 1
    #                 if total_images % 100 == 0:
    #                     print total_images
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_1 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_1 += 1
    #             out2 = net_forward(net,in_stapler)
    #             for ele in zip(out2, val_labels[i-BATCH_SIZE-1:i]):
    #                 # print 'CAT Guess: ',ele[0].argmax(), 'CAT top five choices: ', ele[0].flatten().argsort()[-1:-6:-1] \
    #                 #         , 'CAT correct label: ',ele[1].split()[1]
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_2 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_2 += 1
    #             out3 = net_forward(net,in_pedal)
    #             for ele in zip(out3, val_labels[i-BATCH_SIZE-1:i]):
    #                 if int(ele[0].argmax()) != int(ele[1].split()[1]):
    #                     top_one_3 += 1
    #                 if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
    #                     top_five_3 += 1
    #             input = []
    #         else:
    #             input, i = open_transform_img(transformer, i, caffe_root, input)
    #     print 'total wrong (tribble): ', top_one_1, 'not in top five (tribble): ', top_five_1, 'total images: ', total_images
    #     print 'top-one (tribble) score: ', float(top_one_1)/float(total_images)
    #     print 'top-five (tribble) score: ', float(top_five_1)/float(total_images)
    #     tribble.append((percent_mix,float(top_one_1)/float(total_images),float(top_five_1)/float(total_images)))
    #
    #     print 'total wrong (stapler): ', top_one_2, 'not in top five (stapler): ', top_five_2, 'total images: ', total_images
    #     print 'top-one (stapler) score: ', float(top_one_2)/float(total_images)
    #     print 'top-five (stapler) score: ', float(top_five_2)/float(total_images)
    #     stapler.append((percent_mix,float(top_one_2)/float(total_images),float(top_five_2)/float(total_images)))
    #
    #     print 'total wrong (pedal): ', top_one_3, 'not in top five (pedal): ', top_five_3, 'total images: ', total_images
    #     print 'top-one (pedal) score: ', float(top_one_3)/float(total_images)
    #     print 'top-five (pedal) score: ', float(top_five_3)/float(total_images)
    #     pedal.append((percent_mix,float(top_one_3)/float(total_images),float(top_five_3)/float(total_images)))
    #
    #
    #     # top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    #     # print labels[top_k]
    # print 'unaltered', tribble
    # print 'stapler', stapler
    # print 'pedal', pedal


if __name__ == '__main__':
    main()