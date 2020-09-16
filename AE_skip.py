from vis_data import Visdata # Graph the clustering results

from math import log10, sqrt
import cv2

from keras.models import Model
from keras.layers import Lambda, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Flatten, Add
from keras import optimizers
from keras.models import load_model
from utils import *

from keras.datasets import mnist
import os
from PIL import Image

labels_val = []

BatchSize = 128
Epochs = 200

sum_latent = 0

skip = 0
deno = 0
test = 0
nom = 1

save_path = './save_model/'
graph_save_path = './graph/'
if test == 1:
    en_save = 'Nom_{}_test_{}_encoder'.format(sum_latent,Epochs)
    de_save = 'Nom_{}_test_{}_decoder'.format(sum_latent,Epochs)
    auto_save = 'Nom_{}_test_{}_autoencoder'.format(sum_latent,Epochs)
    graph_name = 'Nom_{}_test_{}_graph'.format(sum_latent,Epochs)
    img_name = 'Nom_{}_test_{}_img'.format(sum_latent,Epochs)
elif skip == 1:
    en_save = 'Nom_{}_skip_{}_encoder'.format(sum_latent,Epochs)
    de_save = 'Nom_{}_skip_{}_decoder'.format(sum_latent, Epochs)
    auto_save = 'Nom_{}_skip_{}_autoencoder'.format(sum_latent,Epochs)
    graph_name = 'Nom_{}_skip_{}_graph'.format(sum_latent,Epochs)
    img_name = 'Nom_{}_skip_{}_img'.format(sum_latent,Epochs)

elif deno == 1:
    en_save = 'Nom_{}_deno_{}_encoder'.format(sum_latent,Epochs)
    de_save = 'Nom_{}_deno_{}_decoder'.format(sum_latent, Epochs)
    auto_save = 'Nom_{}_deno_{}_autoencoder'.format(sum_latent,Epochs)
    graph_name = 'Nom_{}_deno_{}_graph'.format(sum_latent,Epochs)
    img_name = 'Nom_{}_deno_{}_img'.format(sum_latent,Epochs)

else:
    en_save = 'Nom_{}_nom_{}_encoder'.format(sum_latent,Epochs)
    de_save = 'Nom_{}_nom_{}_decoder'.format(sum_latent, Epochs)
    auto_save = 'Nom_{}_nom_{}_autoencoder'.format(sum_latent,Epochs)
    graph_name = 'Nom_{}_nom_{}_graph'.format(sum_latent,Epochs)
    img_name = 'Nom_{}_nom_{}_img'.format(sum_latent,Epochs)


def main():
    global labels_val
    train_new = False
    n_train = 24000
    predict_new = True
    n_predict = 3000
    vis_dim = 2
    build_anim = False

    Img_Path = "C:\\Users\\ialab\\Desktop\\hanja_data\\"
    if deno == 1 :
        save_img_path = 'C:\\Users\\ialab\\Desktop\\hanja_imgs\\hanja_data_deno\\'
    elif skip == 1:
        save_img_path = 'C:\\Users\\ialab\\Desktop\\hanja_imgs\\test_skip\\'
    elif nom == 1:
        save_img_path = 'C:\\Users\\ialab\\Desktop\\hanja_imgs\\test_nom\\'

    x_train, y_train, result = data_set_fun(Img_Path, 0)
    print('y_train :', y_train)
    n_train = len(y_train)
    Img_Path_test = "C:\\Users\\ialab\\Desktop\\hanja_imgs\\noise\\"
    x_test, y_test, result = data_set_fun(Img_Path_test, 0, 1)



    # Build and fit autoencoder
    if train_new:
        autoencoder, encoder, decoder = build_autoencoder((x_train.shape[1],), encoding_dim=30)
        autoencoder.compile(optimizer=optimizers.Adadelta(), loss='mean_squared_error')

        autoencoder.summary()
        decoder.summary()
        autoencoder.fit(x_train[:n_train], x_train[:n_train], epochs=Epochs, batch_size=BatchSize)
        autoencoder.save(save_path + auto_save)
        encoder.save(save_path + en_save)
        decoder.save(save_path + de_save)
    else:
        encoder = load_model(save_path + en_save)
        decoder = load_model(save_path + de_save, custom_objects={'tf': tf})


    # decoder를 가지고  latent code를 이용해 이미지 뽑기
    x_test_0 = np.reshape(x_test[0], (1, 100, 100, 1))
    x_test_1 = np.reshape(x_test[1], (1, 100, 100, 1))
    x_test_0 = encoder.predict(x_test_0)
    x_test_1 = encoder.predict(x_test_1)
    x_test_result = x_test_0 - x_test_1
    print('x_test_0', x_test_0[0])
    print('x_test_1', x_test_1[0])
    print('x_test_result', x_test_result[0])

    decoded_imgs = decoder.predict(x_test_result)


    if test == 1:
        save_images(x_test[:100], image_manifold_size(100), './{}/original.png'.format("skip_connections"))
        save_images(decoded_imgs[:100], image_manifold_size(100), './{}/test_{}.png'.format("skip_connections", Epochs))
        img_name = './{}/test_{}.png'.format("skip_connections", Epochs)
    elif skip == 1:
        save_images(x_test[:100], image_manifold_size(100), './{}/original.png'.format("skip_connections"))
        #save_images(decoded_imgs[:100], image_manifold_size(100), './{}/branch_norm_{}.png'.format("skip_connections", Epochs))
        save_images(decoded_imgs[:100], image_manifold_size(100), './{}/skip_test_{}.png'.format("skip_connections", Epochs))
        img_name = './{}/skip_test_{}.png'.format("skip_connections", Epochs)
        print('save file')
    elif deno == 1:
        save_images(x_test[:100], image_manifold_size(100), './{}/original.png'.format("skip_connections"))
        save_images(decoded_imgs[:100], image_manifold_size(100), './{}/part_deno_{}.png'.format("skip_connections", Epochs))
        img_name = './{}/part_deno_{}.png'.format("skip_connections", Epochs)
    else:
        save_images(x_test[:100], image_manifold_size(100), './{}/original.png'.format("skip_connections"))
        save_images(decoded_imgs[:100], image_manifold_size(100), './{}/nom_{}.png'.format("skip_connections", Epochs))
        img_name = './{}/nom_{}.png'.format("skip_connections", Epochs)


    original = cv2.imread("C:/Users/ialab/Desktop/vis-autoencoder-tsne/skip_connections/original.png")
    compressed = cv2.imread(img_name, 1)
    value = PSNR(original, compressed)
    print(f"PSNR value is {value} dB")


def dataset(images):
    images = images.astype(np.float)
    images = images.reshape(100, 100, 1)

    return images

def data_set_fun(path, set_size, num = 0):
    global labels_val

    filename_list = os.listdir(path)
    if set_size == 0:
        set_size = len(filename_list)

    X_set = np.empty((set_size, 100, 100, 1), dtype=np.float32)
    Y_set = np.empty((set_size), dtype=np.float32)
    name = []

    np.random.shuffle(filename_list)
    result = dict()

    for i, filename in enumerate(filename_list):
        if i >= set_size:
            break

        label = filename.split('.')[0]
        label = label.split('_')[-1]
        result[label] = result.setdefault(label, 0) + 1
        name.append(label)


        file_path = os.path.join(path, filename)
        img = Image.open(file_path)
        img = img.convert('1')  # convert image to black and white
        imgarray = np.array(img)
        imgarray = imgarray.flatten()

        images = dataset(imgarray)

        X_set[i] = images

    if num == 0 :
        labels_val = list(set(name))
        labels_val.sort()
    Y_set = index_label(name)
    return X_set, Y_set, result

def dence_to_one_hot(labels_dence, num_classes):
    num_labes = labels_dence.shape[0]
    index_offset = np.arange(num_labes) * num_classes
    labels_one_hot = np.zeros((num_labes, num_classes))
    labels_one_hot.flat[index_offset + labels_dence.ravel()] = 1  # flat - 배열을 1차원으로 두고 인덱스를 이용해 값 확인
    return labels_one_hot

def index_label(label):

    list = []
    for j in range(len(label)):
        for i in range(len(labels_val)):
            if label[j] == labels_val[i]:
                list.append(i)
                break
    return np.asarray(list)


def import_format_data():
    # Get dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float64') / 255.0
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test = x_test.astype('float64') / 255.0
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(y_train)

    if deno == 1 or test ==1:
        noise_factor = 0.5
        x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)

        return x_train_noisy, y_train, x_test, y_test

    return x_train, y_train, x_test, y_test


def build_autoencoder(input_shape, encoding_dim):
    # Activation function: selu for SNN's: https://arxiv.org/pdf/1706.02515.pdf
    encoding_activation = 'selu'
    decoding_activation = 'selu'
    flatten_layer = Flatten()


    inputs = Input(shape=(100, 100, 1,))
    inputs_de = Input(shape=(encoding_dim,))

    # Encoding layers: successive smaller layers, then a batch normalization layer.
    # Conv1 #
    encoding = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
    encoding = MaxPooling2D(pool_size=(2, 2), padding='same')(encoding)

    if sum_latent == 1:
        z2 = encoding
        z2_2  = flatten_layer(z2)
        z2_2 = Dense(30, activation=encoding_activation)(z2_2)

    # Conv2 #
    encoding = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(encoding)
    encoding = MaxPooling2D(pool_size=(2, 2), padding='same')(encoding)


    # Conv3 #
    encoding = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='same')(encoding)
    encoding = MaxPooling2D(pool_size=(2, 2), padding='same')(encoding)

    if sum_latent == 1:
        z1 = encoding
        z1_2 = flatten_layer(z1)
        z1_2 = Dense(30, activation=encoding_activation)(z1_2)

    encoding = flatten_layer(encoding)  # call it on the given tensor
    feat_dim = 13 * 13 * 8

    encoding = Dense(feat_dim, activation=encoding_activation, kernel_initializer='lecun_normal')(encoding)
    encoding = Dense(int(feat_dim / 2), activation=encoding_activation)(encoding)
    encoding = Dense(encoding_dim, activation=encoding_activation)(encoding)
    sum_encoding = Add()([encoding, z1_2])
    sum_encoding = Add()([sum_encoding, z2_2])





    # Decoding layers for reconstruction
    decoding = Dense(int(feat_dim / 2), activation=decoding_activation)(encoding)
    decoding = Dense(feat_dim, activation=decoding_activation)(decoding)


    decoding = Lambda(lambda x: tf.reshape(x, shape=[-1, 13, 13, 8]))(decoding)


    if skip == 1 :
        decoding = Add()([decoding, z1])



    print('2 :', decoding)

    decoding = Conv2D(8, (3, 3), activation='relu', padding='same')(decoding)  # 4*4*8
    decoding = UpSampling2D((2, 2))(decoding)  # 8*8*8


    decoding = Conv2D(8, (3, 3), activation='relu', padding='same')(decoding)  # 8*8*8
    decoding = UpSampling2D((2, 2))(decoding)  # 16* 16* 8

    decoding = Conv2D(16, (3, 3), activation='relu')(decoding)  # 14* 14* 16


    if skip == 1 or test ==1 :
        decoding = Add()([decoding,z2])


    decoding = UpSampling2D((2, 2))(decoding)

    decoding = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoding)

    # Return the whole model and the encoding section as objects
    autoencoder = Model(inputs, decoding)

    de_2 = autoencoder.layers[-10]
    de_3 = autoencoder.layers[-9]
    de_4 = autoencoder.layers[-8]
    de_5 = autoencoder.layers[-7]
    de_6 = autoencoder.layers[-6]
    de_7 = autoencoder.layers[-5]
    de_8 = autoencoder.layers[-4]
    de_9 = autoencoder.layers[-3]
    de_10 = autoencoder.layers[-2]
    de_11 = autoencoder.layers[-1]


    decoder = Model(inputs_de,de_11(de_10(de_9(de_8(de_7(de_6(de_5(de_4(de_3(de_2(inputs_de)))))))))))


    if sum_latent == 1:
        encoder = Model(inputs, sum_encoding)
    elif sum_latent == 0:
        encoder = Model(inputs, encoding)

    return autoencoder, encoder, decoder





def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr



if __name__ == '__main__':
    main()