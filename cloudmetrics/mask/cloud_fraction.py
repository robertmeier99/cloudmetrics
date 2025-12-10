#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def cloud_fraction(mask):
    """
    Compute cloud fraction for a binary field containing NaNs at its boundaries
    (as it often occurs with geostationary satellite images)

    Parameters
    ----------
    mask : numpy array of shape (npx,npx) - npx is number of pixels
            (cloud) mask field with values 1 (cloud), 0 (no cloud)
            and NaN at the boundaries (lat/lon cut off)

    Returns
    -------
    cf : float
        cloud fraction.

    """
    is_finite = np.isfinite(mask)
    finite_sum = np.sum(is_finite)

    if finite_sum == 0:
        return np.nan
    else:
        return np.sum(mask[is_finite]) / finite_sum
