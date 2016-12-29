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
    data_sets =[[]]
    for img in input:
        i = 0
        for filter in filters:
            combined = (percent_mix*filter) + ((1-percent_mix)*img)
            data_sets[i].append(combined)
            i +=1
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

    print '##################################'
    print 'Resnet'


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
    cat_mixing_image = transformer.preprocess('data', cat_image)
    all_black = np.zeros((cat_mixing_image.shape[1],cat_mixing_image.shape[1],cat_mixing_image.shape[0]))
    black_trans = transformer.preprocess('data', all_black)

    filters = [black_trans]

    #####image mixing
    percents = [0.9]
    black = []
    for percent_mix in percents:
        #image counter
        i = 1

        top_one_2 = 0
        top_five_2 = 0
        total_images = 0
        print 'FOR MIXING PERCENT: ', percent_mix, '\n'
        ##### reading data, altering and predicting and getting accuracy
        input = []
        correct_1 = []
        correct_5 = []
        while i < DATA_SIZE+1:
            if i % BATCH_SIZE == 0:
                input, i = open_transform_img(transformer, i, caffe_root, input)
                # out1 = net_forward(net,input) # give the batch to the net, forward prop and get outputs
                # for ele in zip(out1, val_labels[i-BATCH_SIZE-1:i]):
                #     if total_images % 50 == 0:
                #         print total_images
                #     if int(ele[0].argmax()) != int(ele[1].split()[1]):
                #         top_one_1 += 1
                #     if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
                #         top_five_1 += 1
                in_black = mix_images(input,percent_mix, filters)
                out2 = net_forward(net,in_black)
                for ele in zip(out2, val_labels[i-BATCH_SIZE-1:i]):
                    total_images += 1
                    if total_images % 10 == 0:
                        print total_images
                    # print 'CAT Guess: ',ele[0].argmax(), 'CAT top five choices: ', ele[0].flatten().argsort()[-1:-6:-1] \
                    #         , 'CAT correct label: ',ele[1].split()[1]
                    if int(ele[0].argmax()) != int(ele[1].split()[1]):
                        top_one_2 += 1
                    else:
                        correct_1.append(total_images)
                    if int(ele[1].split()[1]) not in ele[0].flatten().argsort()[-1:-6:-1]:
                        top_five_2 += 1
                    else:
                        correct_5.append(total_images)
                input = []
            else:
                input, i = open_transform_img(transformer, i, caffe_root, input)

        print 'total wrong (black): ', top_one_2, 'not in top five (cat): ', top_five_2, 'total images: ', total_images
        print 'top-one (black) score: ', float(top_one_2)/float(total_images)
        print 'top-five (black) score: ', float(top_five_2)/float(total_images)
        black.append((percent_mix,float(top_one_2)/float(total_images),float(top_five_2)/float(total_images)))
        print 'the correct image numbers are : ', correct_1, 'and', correct_5

        # top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
        # print labels[top_k]
        print 'black', black


if __name__ == '__main__':
    main()