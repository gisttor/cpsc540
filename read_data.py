from collections import defaultdict
import numpy as np
import csv
import PIL
import math
import gc
from multiprocessing.pool import ThreadPool
from keras.utils import np_utils
from keras.preprocessing import image
from keras.utils.generic_utils import Progbar

# convert data into test/validate/train
# format: each business has a set of images

def read_biz_csv(num_labels = 9):
    ''' read dict of businesses:
    structure {biz_id: (binary labels, list of photos)}
    '''
    biz_info = {}
    with open('data/train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)   # skip header
        for row in reader:
            labels = np.zeros(num_labels) - 1
            for c in row[1].split():
                labels[c] = 1
            biz_info[int(row[0])] = (labels, list())

    with open('data/train_photo_to_biz_ids.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)   # skip header
        for row in reader:
            biz_info[int(row[1])][1].append(int(row[0]))

    return biz_info

def get_cache_name(img_size, num_biz):
    return ('data/images_%s-px_%s-biz.npy' % (img_size, num_biz),
            'data/images_%s-px_%s-biz_order.npy' % (img_size, num_biz))

def get_compressed_photos(num_biz, img_size, biz_info = None, 
                          use_cache = True, with_progbar = True):
    ''' Downsamples images into img_size^2 size, and saves to file,
        for a random subset of size num_biz businesses

    Returns:
    tuple of:
    randomized images as a numpy array, shape (num_image, 3, img_size, img_size)
        entries are index, img_channel, img_x, img_y

    numpy array, shape (num_image, 2): vector of (biz_id, img_id) pairs
        corresponding to x
     '''
    if use_cache:
        try:
            x_name, img_id_name = get_cache_name(img_size, num_biz)
            return (np.load(x_name), np.load(img_id_name))
        except IOError:
            pass

    if biz_info is None:
        biz_info = read_biz_csv()

    biz_order = list(biz_info.keys())
    np.random.shuffle(biz_order)

    num_image = 0
    biz_count = 0
    for biz_id in biz_order:
        num_image += len(biz_info[biz_id][1])
        biz_count += 1
        if biz_count >= num_biz:
            break

    # using ones instead of zeros prevents memory from blowing up for some 
    # reason
    # copy on write mechanics?
    print('Reading photos')
    img_ids = np.ones((num_image, 2))
    order = np.random.permutation(num_image)
    biz_count = 0;
    img_count = 0;
    for biz_id in biz_order:
        for photo_id in biz_info[biz_id][1]:
            index = order[img_count]
            img_ids[index] = [biz_id, photo_id]
            img_count += 1
        biz_count += 1
        if biz_count >= num_biz:
            break
    x = read_images(img_size, img_ids[:,1], with_progbar)
    x /= 255;
    if use_cache:
        x_name, img_id_name = get_cache_name(img_size, num_biz)
        np.save(x_name, x)
        np.save(img_id_name, img_ids)
    return x, img_ids

def read_images(img_size, img_list, with_progbar=True):
    num_image = len(img_list)
    if with_progbar:
        progbar = Progbar(num_image)

    x = np.ones((num_image, 3, img_size, img_size), dtype=np.float16)
    for i, photo_id in enumerate(img_list):
        im_raw = image.load_img('data/train_photos/%d.jpg' % photo_id)
        im_raw = im_raw.resize((img_size, img_size), PIL.Image.NEAREST)
        x[i] = image.img_to_array(im_raw)

        if with_progbar:
            progbar.add(1)
    return x

def read_data_photo_labels(test_size,
        img_size = 150, num_labels = 9, test_split = 0.2, batch_size = 32):
    ''' Read data, pushing labels down to photos

    if stream is True, num_biz will be the chunck size, and the first yeild
    will be the test set. Will stream a random subset of businesses each call

    return structure:
    (x_train, y_train, x_test, y_test)
    x_train: image turned to vector
    y_train: categorical labels: binary vectors of size [num classes]
    '''

    biz_info = read_biz_csv(num_labels)
    photo_dict = {}
    # turn into photo_id -> (biz_id, label)
    for biz_id, (labels, photo_list) in biz_info.items():
        for p in photo_list:
            photo_dict[p] = (biz_id, labels)

    def read_with_labels(img_list, prog_bar):
        x = read_images(img_size, img_list, prog_bar)
        y = np.ones((len(img_list), num_labels))
        for i, img in enumerate(img_list):
            y[i] = photo_dict[img][1]
        return x, y

    photo_list = list(photo_dict.keys())
    np.random.shuffle(photo_list)

    x_test, y_test = read_with_labels(photo_list[:test_size], True)
    yield x_test, y_test

    train_list = photo_list[test_size:]

    while True:
        batch = np.random.choice(train_list, size=(batch_size), replace=False)
        yield read_with_labels(batch, False)
        gc.collect()

