from collections import defaultdict
import numpy as np
import csv
import PIL
import gc
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

def get_compressed_photos(num_biz, img_size, biz_info = None):
    ''' Downsamples images into img_size^2 size, and saves to file,
        for a random subset of size num_biz businesses

    Returns:
    tuple of:
    randomized images as a numpy array, shape (num_image, 3, img_size, img_size)
        entries are index, img_channel, img_x, img_y

    numpy array, shape (num_image, 2): vector of (biz_id, img_id) pairs
        corresponding to x
    '''
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
    x = np.ones((num_image, 3, img_size, img_size), dtype=np.int8)
    img_ids = np.ones((num_image, 2))
    order = np.random.permutation(num_image)
    biz_count = 0;
    img_count = 0;
    progbar = Progbar(num_image)
    for biz_id in biz_order:
        for photo_id in biz_info[biz_id][1]:
            im_raw = image.load_img('data/train_photos/%s.jpg' % photo_id)
            im_raw = im_raw.resize((img_size, img_size), PIL.Image.ANTIALIAS)
            index = order[img_count]
            x[index] = image.img_to_array(im_raw)
            img_ids[index] = [biz_id, photo_id]

            img_count += 1
            progbar.add(1)
        biz_count += 1
        if biz_count >= num_biz:
            break

    #x -= 128; 
    x_name, img_id_name = get_cache_name(img_size, num_biz)
    np.save(x_name, x)
    np.save(img_id_name, img_ids)
    return x, img_ids

def read_data_photo_labels(
        num_biz = 100, img_size = 150, num_labels = 9, test_split = 0.1):
    ''' Read data, pushing labels down to photos

    return structure:
    (x_train, y_train, x_test, y_test)
    x_train: image turned to vector
    y_train: categorical labels: binary vectors of size [num classes]
    '''
    
    biz_info = read_biz_csv(num_labels)
    x, img_ids = get_compressed_photos(num_biz, img_size, biz_info)

    # get labels and split into test and train
    n = x.shape[0]
    y = np.ones((n, num_labels))
    for idx in range(n):
        y[idx] = biz_info[img_ids[idx][0]][0]

    split_idx = int(n*(1-test_split))
    x_train, x_test = np.vsplit(x, [split_idx])
    y_train, y_test = np.vsplit(y, [split_idx])
    return (x_train, y_train, x_test, y_test)
