from __future__ import division, unicode_literals
import numpy as np
from matplotlib import pyplot as plt
from numbers import Number
from scipy.optimize import brentq
import logging as lg
import datetime
from inspect import isfunction, isbuiltin, ismethod


J_eV_CONVERSION_JpeV = 1.602e-19


def functionize(func):
    if isinstance(func, Number):
        lg.info('func is a Number: {}.'.format(func))
        return np.vectorize(lambda x: func, otypes=[float])
    if isinstance(func, np.lib.function_base.vectorize):
        pyfunc = getattr(func, 'pyfunc', None)
        call = '.'.join((pyfunc.__module__, pyfunc.__name__))
        lg.info('func is a np.vectorize object: {}'.format(call))
        return func
    elif isinstance(func, np.ufunc):
        call = '.'.join((func.__class__, func.__name__))
        lg.info('func is a np.ufunc object: {}'.format(call))
        return func
    elif isbuiltin(func) and not ismethod(func):
        call_list = []
        mod = getattr(func, '__name__', None)
        if mod:
            call_list.append(mod)
        call_list.append(func.__name__)
        call = '.'.join(call_list)
        lg.info('func is a builtin function: {}'.format(call))
        return np.vectorize(func, otypes=[float])
    elif isfunction(func):
        call = '.'.join((func.__module__, func.__name__))
        lg.info('func is a user-defined function: {}'.format(call))
        return np.vectorize(func, otypes=[float])
    elif ismethod(func):
        call = '.'.join((func.__self__.__name__, func.__self__.__module__, func.__name__))
        lg.info('func is a bound method: {}'.format(call))
        return np.vectorize(func, otypes=[float])
    else:
        raise NotImplementedError('This input function is not yet supported.')


def make_timestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f %z')


def find_roots(func, start, stop, num, plot=False):
    """
    Function to Find All Roots of the Function Between Start and Stop

    :param func:
    :param start:
    :param stop:
    :param num:
    :param plot:
    :return:
    """
    lg.info('find_roots called at {}'.format(make_timestamp()))
    # TODO Support complex
    x = np.linspace(start=start, stop=stop, num=num+2)

    step = x[1] - x[0]

    func = functionize(func)
    eq1 = func(x)

    # Find sign-changes in the difference between the functions, and the indices where sign-changes occur.
    sign_change = np.sign(np.diff(np.sign(eq1)))
    idxs_sign = np.nonzero(sign_change)[0]

    # TODO make sure sign changes don't occur on adjacent samples.
    # sign_durations = np.diff(idxs_sign)

    # Find the slope (difference).
    sign_slope = np.sign(np.diff(eq1))

    roots = []
    roots_fail = []
    roots_abrupt = []

    for idx in idxs_sign:
        root_curr, results = brentq(f=func, a=x[idx] - step, b=x[idx] + step, full_output=True)
        if results.converged:
            if sign_slope[idx-1] == sign_change[idx]:
                roots.append(root_curr)
                lg.info('An Eigenvalue was found'
                        ': {} J = {} eV'.format(root_curr, root_curr/J_eV_CONVERSION_JpeV))
            else:
                lg.warning('An expected vertical asymptote is abruptly considered a root by Brent method.')
                roots_abrupt.append(root_curr)
        else:
            if sign_slope[idx-1] == sign_change[idx]:
                lg.warning('The Brent method failed to find an expected root.')
                roots_fail.append(root_curr)
    return roots, roots_fail, roots_abrupt
