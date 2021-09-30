"""Utilities for EMTF++."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import six
from six.moves import range, zip, map, filter


# ______________________________________________________________________________
# Enums

# Trigger primitives
kDT, kCSC, kRPC, kGEM, kME0 = 0, 1, 2, 3, 4

# ______________________________________________________________________________
# Configs

# The default float type (as defined in tf.keras)
_FLOATX = 'float32'

# The default int type
_INTX = 'int32'

# The fuzz factor used in numeric expressions (as defined in tf.keras)
_EPSILON = 1e-7

# The default image data format convention (as defined in tf.keras)
_IMAGE_DATA_FORMAT = 'channels_last'

# The default fill_value for int type (as defined in numpy.ma)
_MA_FILL_VALUE = 999999

def floatx():
  return _FLOATX

def intx():
  return _INTX

def epsilon():
  return _EPSILON

def image_data_format():
  return _IMAGE_DATA_FORMAT

def ma_fill_value():
  return _MA_FILL_VALUE

# ______________________________________________________________________________
# Functions

def wrap_phi_rad(x):
  # returns phi in [-pi,pi] rad
  twopi = np.pi * 2
  x = x - np.round(x / twopi) * twopi
  return x

def wrap_phi_deg(x):
  # returns phi in [-180.,180] deg
  twopi = 360.
  x = x - np.round(x / twopi) * twopi
  return x

def delta_phi_rad(lhs, rhs):
  x = wrap_phi_rad(lhs - rhs)
  return x

def delta_phi_deg(lhs, rhs):
  x = wrap_phi_deg(lhs - rhs)
  return x

def calc_eta_from_theta_rad(theta_rad):
  eta = -1. * np.log(np.tan(theta_rad / 2.))
  return eta

def calc_eta_from_theta_deg(theta_deg):
  eta = calc_eta_from_theta_rad(np.deg2rad(theta_deg))
  return eta

def calc_theta_rad_from_eta(eta):
  theta = np.arctan2(1.0, np.sinh(eta))  # cot(theta) = sinh(eta)
  return theta

def calc_theta_deg_from_eta(eta):
  theta = np.rad2deg(calc_theta_rad_from_eta(eta))
  return theta

def calc_theta_deg_from_int(theta_int):
  theta = float(theta_int) * (45.0 - 8.5) / 128. + 8.5
  return theta

def calc_theta_rad_from_int(theta_int):
  theta = np.deg2rad(calc_theta_deg_from_int(theta_int))
  return theta

def calc_theta_int(theta, endcap):
  # theta in deg [0..180], endcap [-1, +1]
  theta = 180. - theta if (endcap == -1) else theta
  theta = (theta - 8.5) * 128. / (45.0 - 8.5)
  theta_int = int(round(theta))
  theta_int = 1 if (theta_int <= 0) else theta_int  # protect against invalid value
  return theta_int

def calc_phi_glob_deg_from_loc(loc, sector):
  # loc in deg, sector [1..6]
  glob = loc + 15. + (60. * (sector-1))
  if glob >= 180.:
    glob -= 360.
  return glob

def calc_phi_glob_rad_from_loc(loc, sector):
  # loc in rad, sector [1..6]
  glob = np.deg2rad(calc_phi_glob_deg_from_loc(np.rad2deg(loc), sector))
  return glob

def calc_phi_loc_deg_from_int(phi_int):
  loc = float(phi_int) / 60. - 22.
  return loc

def calc_phi_loc_rad_from_int(phi_int):
  loc = np.deg2rad(calc_phi_loc_deg_from_int(phi_int))
  return loc

def calc_phi_loc_deg_from_glob(glob, sector):
  # glob in deg [-180..180], sector [1..6]
  glob = wrap_phi_deg(glob)
  loc = glob - 15. - (60. * (sector-1))
  return loc

def calc_phi_int(glob, sector):
  # glob in deg [-180..180], sector [1..6]
  loc = calc_phi_loc_deg_from_glob(glob, sector)
  if (loc + 22.) < 0.:
    loc += 360.
  loc = (loc + 22.) * 60.
  phi_int = int(round(loc))
  return phi_int

def calc_ns_from_mhz(mhz):
  return 1e3 / mhz

def calc_mhz_from_ns(ns):
  return 1e3 / ns

def calc_quant_scale(num_bits, num_int_bits):
  return 1.0 / (1 << (num_bits - num_int_bits))

def calc_quant_range(num_bits, num_int_bits, narrow_range=False):
  quant_min = 1 if narrow_range else 0
  quant_max = (1 << num_bits) - 1
  zero_point = (quant_max - quant_min + 1) // 2
  zero_point_from_min = quant_min + zero_point
  range_min = quant_min - zero_point_from_min
  range_max = quant_max - zero_point_from_min
  range_min /= (1 << (num_bits - num_int_bits))
  range_max /= (1 << (num_bits - num_int_bits))
  return (range_min, range_max)

def get_trigger_endsec(endcap, sector):
  # endsec is 0-5 in positive endcap, 6-11 in negative endcap
  assert(endcap == 1 or endcap == -1)
  assert(1 <= sector <= 6)
  result = (sector - 1) if endcap == 1 else (sector - 1 + 6)
  return result

def get_trigger_sector(ring, station, chamber):
  result = np.uint32(0)
  if station > 1 and ring > 1:
    # ch 3-8->1, 9-14->2, ... 1,2 -> 6
    result = ((np.uint32(chamber - 3) & 0x7f) // 6) + 1
  elif station == 1:
    # ch 3-8->1, 9-14->2, ... 1,2 -> 6
    result = ((np.uint32(chamber - 3) & 0x7f) // 6) + 1
  else:
    # ch 2-4->1, 5-7->2, ...
    result = ((np.uint32(chamber - 2) & 0x1f) // 3) + 1
  # max sector is 6, some calculations give a value greater than 6 but this is expected.
  result = np.clip(result, 1, 6)
  return result

def get_trigger_subsector(ring, station, chamber):
  # csc_tp_subsector = (tp_station != 1) ? 0 : ((csc_tp_chamber % 6 > 2) ? 1 : 2);
  result = np.uint32(0)
  if station == 1:
    if np.uint32(chamber) % 6 > 2:
      result = result + 1
    else:
      result = result + 2
  return result

def get_trigger_cscid(ring, station, chamber):
  result = np.uint32(0)
  if station == 1:
    result = np.uint32(chamber) % 3 + 1  # 1,2,3
    if ring == 2:
      result = result + 3
    elif ring == 3:
      result = result + 6
  else:
    if ring == 1:
      result = np.uint32(chamber + 1) % 3 + 1  # 1,2,3
    else:
      result = np.uint32(chamber + 3) % 6 + 4  # 4,5,6,7,8,9
  return result

def get_trigger_cscfr(ring, station, chamber):
  result = np.uint32(0)
  is_overlapping = not (station == 1 and ring == 3)
  is_even = (chamber % 2 == 0)
  if is_overlapping:
    if station < 3:
      result = result + is_even
    else:
      result = result + (not is_even)
  return result

def get_trigger_neighid(ring, station, chamber):
  # neighid is 0 for native chamber, 1 for neighbor chamber
  result = np.uint32(0)
  if station == 1:
    if np.uint32(chamber + 3) % 6 + 1 == 6:
      result = result + 1
  else:
    if ring == 1:
      if np.uint32(chamber + 1) % 3 + 1 == 3:
        result = result + 1
    else:
      if np.uint32(chamber + 3) % 6 + 1 == 6:
        result = result + 1
  return result

def get_trigger_cscneighid(station, subsector, cscid, neighid):
  if neighid == 0:
    return (((subsector - 1) if (station == 1) else 0) * 9) + (cscid - 1);
  else:
    return (((cscid - 1) // 3) + 18) if (station == 1) else (((cscid - 1) >= 3) + 9);

def calc_d0_simple(phi, xv, yv):
  d0 = xv * np.sin(phi) - yv * np.cos(phi)
  return d0

def calc_d0(invpt, phi, xv, yv, B=3.811):
  R = -1.0 / (0.003 * B * invpt)            # R = -pT/(0.003 q B)  [cm]
  xc = xv - (R * np.sin(phi))               # xc = xv - R sin(phi)
  yc = yv + (R * np.cos(phi))               # yc = yv + R cos(phi)
  d0 = R - (np.sign(R) * np.hypot(xc, yc))  # d0 = R - sign(R) * sqrt(xc^2 + yc^2)
  return d0

def calc_etastar_from_eta(invpt, eta, phi, x0, y0, z0, zstar=850., zstar_4T=650.):
  # Propagate to station 2 (z = 850 cm), find r and eta of the track
  # (called rstar and etastar).
  # Note: x0, y0, z0 in cm. Assume pT -> inf.
  if eta < 0:
    zstar *= -1
  # Assume a simplified magnetic field where it is 4T (or 3.811T)
  # inside the solenoid and 0T outside (boundary at z = 650 cm)
  if eta < 0:
    zstar_4T *= -1
  B = 3.811
  R = -1.0 / (0.003 * B * invpt)  # R = -pT/(0.003 q B)  [cm], radius of the circle
  cot = np.sinh(eta)              # cot(theta), which is pz/pt
  if np.abs(zstar_4T) < np.abs(zstar):
    arg_term_4T = np.abs((zstar_4T - z0)/cot)                  # with magfield
    sin_term_4T = (2 * R) * np.sin(arg_term_4T/(2 * R))        # with magfield
    cos_term_4T = (2 * R) * (1 - np.cos(arg_term_4T/(2 * R)))  # with magfield
    arg_term_0T = np.abs((zstar - zstar_4T)/cot)               # without magfield
    sin_term_0T = arg_term_0T                                  # without magfield
    cos_term_0T = 0                                            # without magfield
  else:
    # Also need to check for the boundary at r where 4T -> 0T, ignore for now
    arg_term_4T = np.abs((zstar - z0)/cot)                     # with magfield
    sin_term_4T = (2 * R) * np.sin(arg_term_4T/(2 * R))        # with magfield
    cos_term_4T = (2 * R) * (1 - np.cos(arg_term_4T/(2 * R)))  # with magfield
    arg_term_0T = 0                                            # without magfield
    sin_term_0T = 0                                            # without magfield
    cos_term_0T = 0                                            # without magfield
  phistar_4T = phi + arg_term_4T/(2 * R)  # phi at the boundary where 4T -> 0T
  xstar = x0 + np.cos(phi) * sin_term_4T - np.sin(phi) * cos_term_4T + \
      np.cos(phistar_4T) * sin_term_0T - np.sin(phistar_4T) * cos_term_0T
  ystar = y0 + np.sin(phi) * sin_term_4T + np.cos(phi) * cos_term_4T + \
      np.sin(phistar_4T) * sin_term_0T + np.cos(phistar_4T) * cos_term_0T
  rstar = np.hypot(xstar, ystar)
  cotstar = zstar/rstar
  etastar = np.arcsinh(cotstar)
  return etastar

def calc_signed_rvtx(invpt, eta, phi, x0, y0, z0, zstar=850., zstar_4T=650.):
  # Sign is positive if |etastar| <= |eta|, negative otherwise
  etastar = calc_etastar_from_eta(invpt, eta, phi, x0, y0, z0, zstar, zstar_4T)
  rvtx = np.hypot(x0, y0)
  if not (np.abs(etastar) <= np.abs(eta)):
    rvtx *= -1
  return rvtx

def pick_the_median(lst):  # assume sorted list
  middle = 0 if len(lst) == 0 else (len(lst)-1)//2
  return lst[middle]

def pick_the_first(lst):
  return lst[0]

def find_median_of_three(a0, a1, a2, ma_fill_value=999999):
  vld0 = (a0 != ma_fill_value)
  vld1 = (a1 != ma_fill_value)
  vld2 = (a2 != ma_fill_value)
  median = ma_fill_value
  if vld1 and (not vld0 or a0 <= a1) and (vld2 and a1 <= a2):
    median = a1
  elif vld0 and (not vld2 or a2 <= a0) and (vld1 and a0 <= a1):
    median = a0
  elif vld2 and (not vld1 or a1 <= a2) and (vld0 and a2 <= a0):
    median = a2
  elif vld1 and (not vld2 or not a1 <= a2) and (vld0 and not a0 <= a1):
    median = a1
  elif vld0 and (not vld1 or not a0 <= a1) and (vld2 and not a2 <= a0):
    median = a0
  elif vld2 and (not vld0 or not a2 <= a0) and (vld1 and not a1 <= a2):
    median = a2
  elif vld1 and not vld0 and not vld2:
    median = a1
  elif vld0 and not vld2 and not vld1:
    median = a0
  elif vld2 and not vld1 and not vld0:
    median = a2
  return median

def nan_to_num(a, num=0.0, copy=True):
  a = np.array(a, subok=True, copy=copy)
  mask = np.isnan(a)
  np.copyto(a, num, where=mask)
  return a

def save_np_arrays(outfile, outdict):
  np.savez_compressed(outfile, **outdict)

def stack_np_arrays(lst):
  adict = {}
  outdict = {}

  def _stack_row_splits(lst):
    # Set the first entry to zero.
    new_row_splits = [0]
    for row_splits in lst:
      if not (isinstance(row_splits, (np.ndarray, np.generic)) and
              row_splits.dtype in (np.int64, np.int32) and row_splits.ndim == 1):
        raise TypeError("row_splits must be a 1D int32 or int64 numpy array")
      # Ignore the first entry in row_splits, as the first entry is always zero.
      # Increment all the entries in row_splits by the last value in new_row_splits.
      new_row_splits.extend(new_row_splits[-1] + row_splits[1:])
    new_row_splits = np.asarray(new_row_splits, dtype=np.int32)
    return new_row_splits

  # list of dicts -> dict of lists
  for i, x in enumerate(lst):
    for k, v in six.iteritems(x):
      if i == 0:
        adict[k] = [v]
      else:
        adict[k].append(v)

  # dict of lists -> dict of ndarrays
  for k, v in six.iteritems(adict):
    if k.endswith('_row_splits'):
      new_v = _stack_row_splits(v)
    elif v[0].ndim == 0:
      new_v = np.array(v)
    elif v[0].ndim == 1:
      new_v = np.hstack(v)
    elif v[0].ndim == 2:
      new_v = np.vstack(v)
    elif v[0].ndim == 3:
      new_v = np.dstack(v)
    else:
      new_v = np.concatenate(v, axis=-1)
    outdict[k] = new_v
  return outdict

def hist_digitize(x, bins, ma_fill_value=999999):
  """
  Digitize according to how np.histogram() computes the histogram. All but the last
  (righthand-most) bin is half-open i.e. [a, b). The last bin is closed i.e. [a, b].
  Underflow and overflow values return an index set to `ma_fill_value`.

  Examples:
  --------
  >>> hist_digitize(0, [1,2,3,4])
  array(999999)
  >>> hist_digitize(1, [1,2,3,4])
  array(0)
  >>> hist_digitize(1.1, [1,2,3,4])
  array(0)
  >>> hist_digitize(2, [1,2,3,4])
  array(1)
  >>> hist_digitize(3, [1,2,3,4])
  array(2)
  >>> hist_digitize(4, [1,2,3,4])
  array(2)
  >>> hist_digitize(4.1, [1,2,3,4])
  array(999999)
  >>> hist_digitize(5, [1,2,3,4])
  array(999999)
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
  x = np.ravel(x)
  bin_index = bin_edges.searchsorted(x, side='right')
  bin_index[x == bin_edges[-1]] -= 1
  bin_index[bin_index == 0] = ma_fill_value
  bin_index[bin_index == n_bin_edges] = ma_fill_value
  bin_index[bin_index != ma_fill_value] -= 1
  bin_index = np.squeeze(bin_index)
  return bin_index

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
  bin_index = bin_edges.searchsorted(x, side='right')
  bin_index[bin_index == n_bin_edges] -= 1
  bin_index[bin_index != 0] -= 1
  bin_index = np.squeeze(bin_index)
  return bin_index
