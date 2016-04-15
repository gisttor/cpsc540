from keras.models import Sequential, Graph, model_from_json
from keras.optimizers import SGD
from keras.utils.generic_utils import Progbar

from scipy import misc
import numpy as np
import copy
import glob
import json
import os
from extra_layers import LRN2D

if __name__ == "__main__":
    print("Loading model.")
    # Load model structure
    model = model_from_json(open('googlenet/Keras_model_structure.json').read(), {'LRN2D': LRN2D})
    # Load model weights
    model.load_weights('googlenet/Keras_model_weights.h5') 

    # Compile converted model
    print("Compiling model.")
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    loss = dict()
    for out in model.output_order:
        loss[out] = 'categorical_crossentropy'
    model.compile(optimizer=sgd, loss=loss)
    
    files = glob.glob('data\\train_photos\\*.jpg')
    im_classes = np.zeros((len(files), len(model.output_order), 1000), dtype=np.float32)
    im_names = {}
    # Read images
    progbar = Progbar(len(files))
    for idx, imname in enumerate(files):

        im = misc.imread(imname)

        # Resize
        im = misc.imresize(im, (224, 224)).astype(np.float32)
        # Change RGB to BGR
        aux = copy.copy(im)
        im[:,:,0] = aux[:,:,2]
        im[:,:,2] = aux[:,:,0]
        # Remove train image mean
        im[:,:,0] -= 104.006
        im[:,:,1] -= 116.669
        im[:,:,2] -= 122.679
        # Transpose image dimensions (Keras' uses the channels as the 1st dimension)
        im = np.transpose(im, (2, 0, 1))
        # Insert a new dimension for the batch_size
        im = np.expand_dims(im, axis=0)

        # Load the converted model
        # Predict image output
        in_data = dict()
        for input in model.input_order:
            in_data[input] = im
        pred = model.predict(in_data)

        for j, out_name in enumerate(model.output_order):
            im_classes[idx][j] = pred[out_name]
        filename = os.path.basename(imname)
        im_names[os.path.splitext(filename)[0]] = idx
        progbar.add(1)

    np.save('data/googlenet_predictions1.npy', im_classes)
    with open('data/googlenet_predictions_order1.json', 'w') as order_file:
        json.dump(im_names, order_file)
