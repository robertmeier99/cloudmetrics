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

    return np.sum(mask[is_finite]) / np.sum(is_finite)
