from collections import defaultdict
import numpy as np
import csv
import PIL
import gc
from keras.utils import np_utils
from keras.preprocessing import image
from keras.utils.generic_utils import Progbar

CACHE_XTRAIN = 'x_train.npy'
CACHE_YTRAIN = 'y_train.npy'
CACHE_XTEST  = 'x_test.npy'
CACHE_YTEST  = 'y_test.npy'

# convert data into test/validate/train
# format: each business has a set of images

def load_cache():
    out = []
    for fname in [CACHE_XTRAIN, CACHE_YTRAIN, CACHE_XTEST, CACHE_YTEST]:
        out.append(np.load(fname))
    return tuple(out)

def read_data_photo_labels(
        num_biz = 100, img_size = 150, num_labels = 9, test_split = 0.1,
        fromfile = True):
    ''' Read data, pushing labels down to photos

    return structure:
    (x_train, y_train, x_test, y_test)
    x_train: image turned to vector
    y_train: categorical labels: binary vectors of size [num classes]
    '''
    if fromfile:
        try:
            return load_cache()
        except IOError:
            pass
    
    # construct dict of businesses:
    # structure {biz_id: (binary labels, list of photos)}
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

    # shape: num_biz, 3 channels, img x, img y
    num_image = 0
    biz_count = 0
    for biz, (labels, photo_ids) in biz_info.items():
        num_image += len(photo_ids)
        biz_count += 1
        if biz_count >= num_biz:
            break

    print('Reading photos')
    # using ones instead of zeros prevents memory from blowing up for some 
    # reason
    # copy on write mechanics?
    num_test = int(num_image*test_split)
    num_train = num_image - num_test
    x_train = np.ones((num_train, 3, img_size, img_size), dtype=np.float32)
    y_train = np.ones((num_train, num_labels), dtype=np.int32)
    x_test = np.ones((num_test, 3, img_size, img_size), dtype=np.float32)
    y_test = np.ones((num_test, num_labels), dtype=np.int32)
    biz_count = 0;
    img_count = 0;
    order = np.random.permutation(num_image)
    progbar = Progbar(num_image)
    for biz, (labels, photo_ids) in biz_info.items():
        for photo_id in photo_ids:
            im_raw = image.load_img('data/train_photos/%s.jpg' % photo_id)
            im_raw = im_raw.resize((img_size, img_size), PIL.Image.ANTIALIAS)
            index = order[img_count]
            if index < num_train:
                x_train[index] = image.img_to_array(im_raw)
                y_train[index] = labels
            else:
                x_test[index - num_train] = image.img_to_array(im_raw)
                y_test[index - num_train] = labels
            img_count += 1
            progbar.add(1)
        biz_count += 1
        if biz_count >= num_biz:
            break
    
    x_train /= 255; 
    x_test /= 255; 
    
    for fname, arr in \
        zip([CACHE_XTRAIN, CACHE_YTRAIN, CACHE_XTEST, CACHE_YTEST],
            [x_train, y_train, x_test, y_test]):
       np.save(fname, arr)

    return (x_train, y_train, x_test, y_test)

