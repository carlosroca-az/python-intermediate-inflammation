"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np


class Observation:
    def __init__(self, day, value):
        self.day = day
        self.value = value

    def __str__(self):
        return str(self.value)


class Person:
    """A person."""
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


class Patient(Person):
    """A patient in an inflammation study."""
    def __init__(self, name, observations=None):
        super().__init__(name)

        self.observations = []
        if observations is not None:
            self.observations = observations

    def add_observation(self, value, day=None):
        if day is None:
            try:
                day = self.observations[-1].day + 1
            except IndexError:
                day = 0
        new_observation = Observation(day, value)
        self.observations.append(new_observation)
        return new_observation


class Doctor(Person):
    """A doctor in an inflammation study."""
    def __init__(self, name):
        super().__init__(name)
        self.patients = []

    def add_patient(self, new_patient):
        # A crude check by name if this patient is already looked after
        # by this doctor before adding them
        for patient in self.patients:
            if patient.name == new_patient.name:
                return
        self.patients.append(new_patient)


def load_csv(filename):
    """Load a Numpy array from a CSV

    :param filename: Filename of CSV to load
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.

    :param data: matrix with inflammation data (rows: patients, columns: days)
    :returns: array with daily std devs of inflammation for each patient
    """
    return np.mean(data, axis=0)


def daily_stddev(data):
    """Calculate the daily standard deviation of a 2D inflammation data array.

    :param data: matrix with inflammation data (rows: patients, columns: days)
    :returns: array with daily means of inflammation for each patient
    """
    return np.std(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.

    :param data: matrix with inflammation data (rows: patients, columns: days)
    :returns: array with maximum value of inflammation for each patient
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.

    :param data: matrix with inflammation data (rows: patients, columns: days)
    :returns: array with minimum value of inflammation for each patient
    """
    return np.min(data, axis=0)


def patient_normalise(data):
    """
    Normalise patient data from a 2D inflammation data array.

    NaN values are ignored, and normalised to 0.
    Negative values raise a ValueError exception.

    :param data: 2D array of inflammation data
    :type data: ndarray
    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data input should be ndarray')
    elif len(data.shape) != 2:
        raise ValueError('inflammation array should be 2D')
    elif np.any(data < 0):
        raise ValueError('inflammation values should be non-negative')

    max_data = np.nanmax(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / max_data[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    return normalised
