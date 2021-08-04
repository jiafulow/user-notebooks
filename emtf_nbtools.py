"""Notebook tools for EMTF++."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import types


def get_logger():
  import os
  import logging

  # Copied from https://docs.python.org/2/howto/logging.html
  # create logger
  cwd = os.path.basename(os.getcwd())
  logger = logging.getLogger(cwd)
  logger.setLevel(logging.DEBUG)

  # create file handler which logs even debug messages
  fh = logging.FileHandler(cwd+'.log')
  fh.setLevel(logging.DEBUG)

  # create console handler with a higher log level
  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)

  # create formatter and add it to the handlers
  formatter = logging.Formatter('%(asctime)s [%(levelname)-8s] %(message)s')
  fh.setFormatter(formatter)
  formatter = logging.Formatter('[%(levelname)-8s] %(message)s')
  ch.setFormatter(formatter)

  # add the handlers to the logger
  if not len(logger.handlers):
    logger.addHandler(fh)
    logger.addHandler(ch)
  return logger


def get_colormap():
  from matplotlib import cm
  from matplotlib.colors import ListedColormap, LinearSegmentedColormap

  viridis_mod = ListedColormap(cm.viridis.colors, name='viridis_mod')
  viridis_mod.set_under('w', 1)

  cdict = {
    'red'  : ((0.0, 0.0, 0.0), (0.746032, 0.0, 0.0), (1.0, 1.0, 1.0)),
    'green': ((0.0, 0.0, 0.0), (0.365079, 0.0, 0.0), (0.746032, 1.0, 1.0), (1.0, 1.0, 1.0)),
    'blue' : ((0.0, 0.0416, 0.0416), (0.365079, 1.0, 1.0), (1.0, 1.0, 1.0)),
  }
  blue_hot = LinearSegmentedColormap('blue_hot', cdict)

  cdict = {
    'red'  : ((0.0, 0.0, 0.0), (0.746032, 0.0, 0.0), (1.0, 1.0, 1.0)),
    'green': ((0.0, 0.0416, 0.0416), (0.365079, 1.0, 1.0), (1.0, 1.0, 1.0)),
    'blue' : ((0.0, 0.0, 0.0), (0.365079, 0.0, 0.0), (0.746032, 1.0, 1.0), (1.0, 1.0, 1.0)),
  }
  green_hot = LinearSegmentedColormap('green_hot', cdict)

  cdict = {
    'red'  : ((0.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    'green': ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
    'blue' : ((0.0, 1.0, 1.0), (1.0, 0.0, 0.0)),
  }
  red_binary = LinearSegmentedColormap('red_binary', cdict)

  # Update 'cm'
  cm.viridis_mod = viridis_mod
  cm.blue_hot = blue_hot
  cm.green_hot = green_hot
  cm.red_binary = red_binary
  return cm
