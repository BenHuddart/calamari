"""Fixtures shared across these behavioural tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    """Independent noise stream for each test; never advanced by another test."""
    return np.random.default_rng(20260828)
