import math
import sys

from keras import backend as K
from keras.engine import Layer, InputSpec

import tensorflow as tf

import cv2

from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed

import numpy as np
import random
import copy

from keras.objectives import categorical_crossentropy

import time

import pickle

from matplotlib import pyplot as plt
from keras.models import Model

import os
import pandas as pd

from keras.optimizers import Adam, SGD, RMSprop

from keras.utils import generic_utils
