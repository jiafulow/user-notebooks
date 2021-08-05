"""Notebook tools for EMTF++."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import functools
import types

from scipy.optimize import curve_fit
from scipy.stats import beta


def get_logger():
  import logging

  # Copied from https://docs.python.org/2/howto/logging.html
  # create logger
  logger = logging.getLogger('notebook')
  logger.setLevel(logging.DEBUG)

  # create file handler which logs even debug messages
  fh = logging.FileHandler('notebook.log')
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


# Create a module within a module
emtf_nbtools = types.ModuleType('emtf_nbtools')


def export_to_emtf_nbtools(name):
  """A decorator with argument that adds a fn to the emtf_nbtools module."""

  def decorator(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
      return fn(*args, **kwargs)

    setattr(emtf_nbtools, name, wrapper)
    return wrapper

  return decorator


@export_to_emtf_nbtools('gaus')
def gaus(x, a, mu, sig):
  return a * np.exp(-0.5 * np.square((x - mu) / sig))


@export_to_emtf_nbtools('fit_gaus')
def fit_gaus(hist, edges, mu=0.0, sig=1.0):
  hist = hist.astype(np.float64)
  edges = edges.astype(np.float64)
  xdata = (edges[1:] + edges[:-1]) / 2
  ydata = hist
  popt, pcov = curve_fit(gaus, xdata, ydata, p0=[np.max(hist), mu, sig])
  if not np.isfinite(pcov).all():
    raise RuntimeError('Fit has failed to converge.')
  popt[2] = np.abs(popt[2])  # take absolute value of sigma
  return popt


@export_to_emtf_nbtools('mean_squared_error')
def mean_squared_error(y_true, y_pred):
  return np.square(y_pred - y_true).mean(axis=-1)


@export_to_emtf_nbtools('mean_absolute_error')
def mean_absolute_error(y_true, y_pred):
  return np.abs(y_pred - y_true).mean(axis=-1)


@export_to_emtf_nbtools('median_absolute_deviation')
def median_absolute_deviation(y_true, y_pred):
  scale = 0.6744897501960817  # special.ndtri(0.75)
  return np.median(np.abs(y_pred - y_true), axis=-1) / scale


@export_to_emtf_nbtools('find_sumw2_errors')
def find_sumw2_errors(y, w):
  sumw2 = y * np.square(w)
  return np.sqrt(sumw2)


@export_to_emtf_nbtools('find_efficiency_errors')
def find_efficiency_errors(total_array, passed_array, level=0.682689492137):
  """Copied from ROOT TEfficiency::ClopperPearson()."""

  if isinstance(total_array, list):
    total_array = np.array(total_array)
  if isinstance(passed_array, list):
    passed_array = np.array(passed_array)
  assert total_array.ndim == 1
  assert passed_array.ndim == 1

  alpha = (1. - level) / 2
  lower_array = np.zeros(total_array.shape[0], dtype='float64')
  upper_array = np.zeros(total_array.shape[0], dtype='float64')

  for i, (total, passed) in enumerate(zip(total_array, passed_array)):
    if total == 0.:
      eff = 0.
    else:
      eff = np.true_divide(passed, total)
    if passed == 0.:
      lo = 0.
    else:
      lo = beta.ppf(alpha, passed, total - passed + 1)
    if passed == total:
      up = 1.
    else:
      up = beta.ppf(1. - alpha, passed + 1, total - passed)
    lower_array[i] = eff - lo
    upper_array[i] = up - eff
  return np.vstack((lower_array, upper_array))


@export_to_emtf_nbtools('hist_digitize_inclusive')
def hist_digitize_inclusive(x, bins):
  """
  Digitize according to how np.histogram() computes the histogram. All but the last
  (righthand-most) bin is half-open i.e. [a, b). The last bin is closed i.e. [a, b].
  Underflow values return an index of 0, overflow values return an index of len(bins)-2.

  Examples:
  --------
  >>> hist_digitize_inclusive(0, [1,2,3,4])
  array(0)
  >>> hist_digitize_inclusive(1, [1,2,3,4])
  array(0)
  >>> hist_digitize_inclusive(1.1, [1,2,3,4])
  array(0)
  >>> hist_digitize_inclusive(2, [1,2,3,4])
  array(1)
  >>> hist_digitize_inclusive(3, [1,2,3,4])
  array(2)
  >>> hist_digitize_inclusive(4, [1,2,3,4])
  array(2)
  >>> hist_digitize_inclusive(4.1, [1,2,3,4])
  array(2)
  >>> hist_digitize_inclusive(5, [1,2,3,4])
  array(2)
  """
  bin_edges = np.asarray(bins)
  n_bin_edges = bin_edges.size
  if n_bin_edges < 2:
    raise ValueError('`bins` must have size >= 2')
  if bin_edges.ndim != 1:
    raise ValueError('`bins` must be 1d')
  if np.any(bin_edges[:-1] > bin_edges[1:]):
    raise ValueError('`bins` must increase monotonically')
  x = np.asarray(x)
  x = x.ravel()
  # bin_index has range 0..len(bins)
  bin_index = bin_edges.searchsorted(x, side='right')
  # modified bin_index has range 0..len(bins)-2
  bin_index[bin_index == n_bin_edges] -= 1
  bin_index[bin_index != 0] -= 1
  bin_index = np.squeeze(bin_index)
  return bin_index


class _BinProxy(object):
  def __init__(self, hist, idx):
    self._hist = hist
    self._idx = idx

  @property
  def value(self):
    return self._hist.get_bin_content(self._idx)

  @value.setter
  def value(self, v):
    return self._hist.set_bin_content(self._idx, v)

  @property
  def error(self):
    return self._hist.get_bin_error(self._idx)

  @error.setter
  def error(self, e):
    return self._hist.set_bin_error(self._idx, e)


@export_to_emtf_nbtools('DumbHist')
class DumbHist(object):
  def __init__(self, edges, dtype='float64', err_dtype='float64'):
    self._edges = edges
    # Unlike the ROOT TH1, there are no underflow or overflow bins
    self._bin_content = np.zeros(len(edges) - 1, dtype=dtype)
    self._bin_error = np.zeros(len(edges) - 1, dtype=err_dtype)

  def __len__(self):
    return len(self._bin_content)

  def __iter__(self):
    for i in range(len(self)):
      bproxy = _BinProxy(self, i)
      yield bproxy

  def __reversed__(self):
    for i in reversed(range(len(self))):
      bproxy = _BinProxy(self, i)
      yield bproxy

  def get_bin_content(self, index):
    return self._bin_content[index]

  def set_bin_content(self, index, value):
    self._bin_content[index] = value

  def get_bin_error(self, index):
    return self._bin_error[index]

  def set_bin_error(self, index, error):
    self._bin_error[index] = error

  def __getitem__(self, index):
    if isinstance(index, slice):
      raise TypeError('index cannot be an instance of slice')
    if isinstance(index, tuple):
      raise TypeError('index cannot be an instance of tuple')
    return _BinProxy(self, index)

  def __setitem__(self, index, value):
    if isinstance(index, slice):
      raise TypeError('index cannot be an instance of slice')
    if isinstance(index, tuple):
      raise TypeError('index cannot be an instance of tuple')
    if isinstance(value, _BinProxy):
      self.set_bin_content(index, value.value)
      self.set_bin_error(index, value.error)
    elif isinstance(value, tuple):
      value, error = value
      self.set_bin_content(index, value)
      self.set_bin_error(index, error)
    else:
      self.set_bin_content(index, value)

  def find_bin(self, x):
    index = self.edges.searchsorted(x, side='right')
    if index == len(self):
      index -= 1
    if index != 0:
      index -= 1
    return index

  def scale(self, factor):
    self._bin_content = factor * self._bin_content
    self._bin_error = factor * self._bin_error

  @property
  def edges(self):
    return self._edges

  @property
  def dtype(self):
    return self._bin_content.dtype

  @property
  def err_dtype(self):
    return self._bin_error.dtype

  @property
  def steps(self):
    return np.append(self._bin_content, 0)

  @property
  def err_steps(self):
    return np.append(self._bin_error, 0)
