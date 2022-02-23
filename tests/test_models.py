"""Tests for statistics functions within the Model layer."""

import numpy as np
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
        ([[1, 2], [3, 4], [5, 6]], [1.633, 1.633],  # positive integers
         [[1, 2], [3, 4], [5, float('nan')]], [1.633, float('nan')]),  # positive ints and nan
    ]
)
def test_daily_stddev(test, expected):
    """Test that the standard deviation function works for cases above."""
    from inflammation.models import daily_stddev
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_almost_equal(daily_stddev(test), expected, decimal=2)


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


@pytest.mark.parametrize(
    "test, expected, raises",
    [
        ('hi there',
         None,
         TypeError),
        (3.1416,
         None,
         TypeError),
        ([1, 2, 3, 4, 5, 6],
         None,
         ValueError),
        ([[-1, 2, 3], [-4, 5, 6], [7, 8, 9]],
         [[0, 0.67, 1], [0, 0.83, 1], [0.78, 0.89, 1]],
         ValueError),
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
         None),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]],
         [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
         None),
        ([[1, float('nan'), 3], [4, 5, 6], [7, 8, float('nan')]],
         [[0.33, 0, 1], [0.67, 0.83, 1], [0.88, 1, 0]],
         None),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
         [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
         None)
    ])
def test_patient_normalise(test, expected, raises):
    """Test normalisation works for arrays of one and positive integers."""
    from inflammation.models import patient_normalise
    if isinstance(test, list):
        test = np.array(test)
    if raises:
        with pytest.raises(raises):
            npt.assert_almost_equal(patient_normalise(test), np.array(expected), decimal=2)
    else:
        npt.assert_almost_equal(patient_normalise(test), np.array(expected), decimal=2)
