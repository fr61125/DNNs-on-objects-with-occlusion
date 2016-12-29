import matplotlib.pyplot as plt
import numpy as np
import caffe
import sys


caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')
caffe.set_mode_cpu()


def ref_net_img_processing(net, caffe_root):
    mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    return transformer


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

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
net = caffe.Net(model_def,model_weights,caffe.TEST)
transformer = ref_net_img_processing(net, caffe_root)

test_image = caffe.io.load_image(caffe_root + 'examples/images/ILSVRC2012_val_00000001.jpeg')
mixing_image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')

transformed_mixing_image = transformer.preprocess('data', mixing_image)
transformed_test_image = transformer.preprocess('data', test_image)

noise, gauss = noisy(mixing_image)
trans_noise = transformer.preprocess('data', noise)
trans_gauss = transformer.preprocess('data', gauss)

all_white = np.ones((transformed_mixing_image.shape[1],transformed_mixing_image.shape[1],transformed_mixing_image.shape[0]))
white_trans = transformer.preprocess('data', all_white)
all_black = np.zeros((transformed_mixing_image.shape[1],transformed_mixing_image.shape[1],transformed_mixing_image.shape[0]))
black_trans = transformer.preprocess('data', all_black)

combined = trans_noise*0.3 + transformed_mixing_image*0.7
mixed = transformer.deprocess('data', combined)
plt.imshow(mixed)
plt.show()
combined = trans_gauss + transformed_mixing_image
mixed = transformer.deprocess('data', combined)
plt.imshow(mixed)
plt.show()