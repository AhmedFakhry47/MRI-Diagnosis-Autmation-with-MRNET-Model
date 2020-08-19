import matplotlib.pyplot as plt
import numpy as np
import os
from imgaug import augmenters as iaa
import imgaug as ia
import gc
import csv
import keras
import pandas as pd
from keras.models import Sequential
import csv
from keras.layers import GlobalMaxPooling1D, Dense, Conv1D,Conv2D
from keras import regularizers,optimizers
import keras,os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Flatten,
    merge,
    Lambda
)
from __future__ import print_function

import numpy as np
import warnings
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.engine.topology import get_source_inputs
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D,
    ZeroPadding2D
)
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

from __future__ import print_function
from __future__ import absolute_import

import warnings
import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image

# Evidently this model breaks Python's default recursion limit
# This is a theano issue
import sys
sys.setrecursionlimit(10000)

from keras.models import Model