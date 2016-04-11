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

def read_data_photo_labels(num_biz = 100, test_ratio = 0.1, img_size = 150, num_labels = 9):
    ''' Read data, pushing labels down to photos

    return structure:
    x_train, y_train
    x_train: image turned to vector
    y_train: categorical labels: binary vectors of size [num classes]
    '''
    # construct dict of businesses:
    # structure {biz_id: (binary labels, list of photos)}
    biz_info = {}
    with open('data/train.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)   # skip header
        for row in reader:
            labels = np.zeros(num_labels)
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
    biz_count = 0;
    for biz, (labels, photo_ids) in biz_info.items():
        num_image += len(photo_ids)
        biz_count += 1
        if biz_count >= num_biz:
            break

    print('Reading photos')
    # using ones instead of zeros prevents memory from blowing up for some reason
    x = np.ones((num_image, 3, img_size, img_size), dtype=np.float32)
    y = np.ones((num_image, num_labels), dtype=np.int32)
    biz_count = 0;
    img_count = 0;
    order = np.random.permutation(num_image) - 1
    progbar = Progbar(num_image)
    for biz, (labels, photo_ids) in biz_info.items():
        for photo_id in photo_ids:
            im_raw = image.load_img('data/train_photos/%s.jpg' % photo_id)
            im_raw = im_raw.resize((img_size, img_size), PIL.Image.NEAREST)
            x[order[img_count]] = image.img_to_array(im_raw)
            y[order[img_count]] = labels
            img_count += 1
            progbar.add(1)
        biz_count += 1
        gc.collect()
        if biz_count >= num_biz:
            break
    x = x/255; 
    return x,y

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
