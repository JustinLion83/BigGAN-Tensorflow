import scipy.misc
import numpy as np
import os
from glob import glob

import tensorflow as tf
import tensorflow.contrib.slim as slim
from keras.datasets import cifar10, mnist

class ImageData:

    def __init__(self, load_size, channels, custom_dataset):
        self.load_size = load_size
        self.channels = channels
        self.custom_dataset = custom_dataset

    def image_processing(self, filename):
    # 圖片前處理, 將圖片resize到[load_size, load_size] -> 接著把pixel_value 縮放到[-1.0, 1.0]
        if not self.custom_dataset :
            x_decode = filename
        else :
            x = tf.read_file(filename)
            x_decode = tf.image.decode_jpeg(x, channels=self.channels)

        img = tf.image.resize_images(x_decode, [self.load_size, self.load_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        return img


def load_mnist():
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()
    x = np.concatenate((train_data, test_data), axis=0)
    x = np.expand_dims(x, axis=-1)

    return x

def load_cifar10() :
    (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
    x = np.concatenate((train_data, test_data), axis=0)

    return x

def load_data(dataset_name) :
    # 看是要載入甚麼資料庫 mnist 或 cifar10 或 自己的Dataset
    if dataset_name == 'mnist' :
        x = load_mnist()
    elif dataset_name == 'cifar10' :
        x = load_cifar10()
    else :

        x = glob(os.path.join("./dataset", dataset_name, '*.*'))

    return x


def preprocessing(x, size):
    # 可用cv2取代
    x = scipy.misc.imread(x, mode='RGB')
    x = scipy.misc.imresize(x, [size, size])
    x = normalize(x)
    return x

def normalize(x) :
    return x/127.5 - 1

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def merge(images, size):
    '''把output_img合併成大圖並輸出(我有自己的方法)'''
    # 檢查最後的channel是 3或4或1
    h, w = images.shape[1], images.shape[2]
    # 彩色圖
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c)) # 全黑的大圖
        for idx, image in enumerate(images):
            # 記憶點: 是否"連續變動" (Continuous Change)
            i = idx % size[1]     # column (每次都+1)        連續變動
            j = idx // size[1]    # row    (每size次才會+1)  非連續變動
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    
    # 灰度圖
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
    # image = np.squeeze(merge(images, size)) # 채널이 1인거 제거 ?
    return scipy.misc.imsave(path, merge(images, size))


def inverse_transform(images):
    return (images+1.)/2.


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def str2bool(x):
    return x.lower() in ('true')

##################################################################################
# Regularization    正交(垂直)初始化
##################################################################################

# 用在卷積
def orthogonal_regularizer(scale) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Conv layer as kernel regularizer"""

    def ortho_reg(w) :
        """ Reshaping the matrxi in to 2D tensor for enforcing orthogonality"""
        _, _, _, c = w.get_shape().as_list()
        # 把batch_size, w, z 都壓縮成一個數字 
        w = tf.reshape(w, [-1, c])

        """ Declaring a Identity Tensor of appropriate size"""
        # 對角線1, 其他元素為0
        identity = tf.eye(c)

        """ Regularizer Wt*W - I """
        # 移除w(權重)對角線上的元素
        w_transpose = tf.transpose(w)       # W的轉置(Wt)
        w_mul = tf.matmul(w_transpose, w)   # Wt*W
        reg = tf.subtract(w_mul, identity)  # Wt*W - I

        """Calculating the Loss Obtained"""
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg

# 用在Dense
def orthogonal_regularizer_fully(scale) :
    """ Defining the Orthogonal regularizer and return the function at last to be used in Fully Connected Layer """

    def ortho_reg_fully(w) :
        """ Reshaping the matrix in to 2D tensor for enforcing orthogonality"""
        _, c = w.get_shape().as_list()

        """Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """ Calculating the Loss """
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg_fully
