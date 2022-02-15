"""Tests for statistics functions within the Model layer."""

import numpy.testing as npt
import pytest


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),     # zeros
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),     # positive integers
    ]
)
def test_daily_mean(test, expected):
    """Test that mean function works for cases above."""
    from inflammation.models import daily_mean
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test), expected)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),     # zeros
        ([[1, 2], [3, 4], [5, 6]], [5, 6]),     # positive integers
    ]
)
def test_daily_max(test, expected):
    """Test that max function works for cases above."""
    from inflammation.models import daily_max
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(test), expected)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),     # zeros
        ([[1, 2], [3, 4], [5, 6]], [1, 2]),     # positive integers
    ]
)
def test_daily_min(test, expected):
    """Test that min function works for cases above."""
    from inflammation.models import daily_min
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_min(test), expected)


def test_daily_min_string():
    """Test for TypeError when passing strings"""
    from inflammation.models import daily_min

    with pytest.raises(TypeError):
        daily_min([['Hello', 'there'], ['General', 'Kenobi']])
