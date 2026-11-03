"""
Private helpers for writing pvlib functions against the Python array API
standard, https://data-apis.org/array-api/latest/

Model code is written once against an array namespace ``xp`` and then runs
on any conforming array library (NumPy, CuPy, PyTorch, JAX, Dask, ...).

The contributing guide shows how to port a function and explains the design:
docs/sphinx/source/contributing/array_libraries.rst and
array_libraries_design.rst.

Cores and boundaries
--------------------
Each ported function is a core: it takes the namespace as ``xp`` and assumes
its array arguments are already arrays of it, so it is only the model's
math. :py:func:`labeled_io` makes the core into the public function, the
boundary, which once per call unwraps pandas and xarray objects (they do not
implement the standard), finds the namespace, converts the arguments, and
wraps the result again. pandas in, pandas out (and xarray in, xarray out) is
unchanged for users.

pvlib functions call each other's cores, ``aoi_projection.core(..., xp=xp)``,
so a call that goes through several functions crosses the boundary once.
``tools._cosd`` and the other private trig helpers take ``xp`` for the same
reason.

Time
----
The standard has no datetime dtype, so time enters the array core as a
number: the quantity the model actually uses. ``labeled_io`` converts
datetime-like arguments (DatetimeIndex, datetime64 arrays and Series,
Timestamps, xarray time coordinates) at the boundary to one of

- ``'unix'``: float64 seconds since 1970-01-01 00:00 UTC
- ``'doy'``: day of year, UTC

and Timedeltas to float seconds. Inside the core, :py:func:`unix_to_civil`
recovers calendar fields and :py:func:`rolling_sum` does time-window
aggregation, on any array library.
"""

import contextlib
import datetime
import functools
import inspect
import math

import numpy as np
import pandas as pd
import array_api_compat
import array_api_compat.numpy as np_compat
from array_api_compat import is_array_api_obj, is_lazy_array


_SCALAR_TYPES = (bool, int, float, complex, np.generic)
_NUMPY_TYPES = _SCALAR_TYPES + (np.ndarray, list, tuple, pd.Series,
                                pd.DataFrame)


def array_namespace(*xs):
    """
    Array namespace for the inputs ``xs``.

    ``None``, Python and NumPy scalars, lists, tuples and pandas objects are
    treated as NumPy. NumPy arrays join a Dask calculation, since Dask arrays
    are NumPy arrays computed lazily (xarray mixes them the same way). Other
    inputs from two different array libraries raise ``TypeError``.
    """
    # fast path for the common case, see the benchmarks in GH 2515
    if all(x is None or isinstance(x, _NUMPY_TYPES) for x in xs):
        return np_compat
    arrays = [x for x in xs if is_array_api_obj(x)
              and not isinstance(x, np.generic)]
    if not arrays:
        return np_compat
    if any(array_api_compat.is_dask_array(x) for x in arrays):
        arrays = [x for x in arrays if not isinstance(x, np.ndarray)]
    return array_api_compat.array_namespace(*arrays)


def asarrays(*xs, xp):
    """
    Convert ``xs`` to floating-point arrays with a common dtype and device.

    The dtype is the promoted dtype of the floating-point array inputs, or
    float64 if there are none. The device is that of the largest array, so a
    0-d CPU tensor does not pull GPU data to the CPU. Masked values of NumPy
    masked arrays become NaN, so their fill values never enter a calculation
    (:py:func:`labeled_io` restores the mask). ``None`` passes through.
    """
    if xp is np_compat:
        floats = [x.dtype for x in xs
                  if isinstance(x, np.ndarray) and x.dtype.kind == 'f']
        dtype = np.result_type(*floats) if floats else np.float64
        return tuple(None if x is None else _np_asarray(x, dtype)
                     for x in xs)
    arrays = [x for x in xs if is_array_api_obj(x)
              and not isinstance(x, np.generic)]
    floats = [x for x in arrays if xp.isdtype(x.dtype, 'real floating')]
    dtype = xp.result_type(*floats) if floats else xp.float64
    device = (array_api_compat.device(max(arrays, key=_size)) if arrays
              else None)
    return tuple(None if x is None else
                 xp.asarray(x, dtype=dtype, device=device) for x in xs)


def _np_asarray(x, dtype):
    if isinstance(x, np.ma.MaskedArray):
        return np.ma.filled(x.astype(dtype), np.nan)
    return np.asarray(x, dtype=dtype)


def _size(x):
    # None for lazy arrays of unknown size, which are likely large
    n = array_api_compat.size(x)
    return math.inf if n is None else n


def is_concrete(x):
    """
    True if the values of ``x`` can be read now, so data-dependent checks
    can run: not a JAX tracer (inside ``jax.jit``) or a lazy Dask array.
    """
    if array_api_compat.is_jax_array(x):
        import jax
        return not isinstance(x, jax.core.Tracer)
    return not is_lazy_array(x)


def known_any(cond, xp):
    """
    ``xp.any(cond)`` as a bool, or False if its value isn't known yet. For
    input checks that need values: they are skipped inside ``jax.jit``,
    where anything computed is traced, even from arrays made outside, and
    for lazy Dask arrays, whose evaluation a check should not force.
    """
    result = xp.any(cond)
    return is_concrete(result) and bool(result)


def errstate(xp, **kwargs):
    """``np.errstate`` for NumPy, a no-op for other libraries."""
    if array_api_compat.is_numpy_namespace(xp):
        return np.errstate(**kwargs)
    return contextlib.nullcontext()


def deg2rad(x):
    # np.radians multiplies by pi/180, so results match it bit for bit
    if isinstance(x, (list, tuple)):
        x = np.asarray(x)
    return x * (np.pi / 180)


def rad2deg(x):
    if isinstance(x, (list, tuple)):
        x = np.asarray(x)
    return x * (180 / np.pi)


def _is_dataarray(x):
    # avoid importing xarray unless it is already in use
    cls = type(x)
    return cls.__name__ == 'DataArray' and cls.__module__.startswith('xarray')


# --- time at the boundary ---------------------------------------------------

_DATETIME_SCALARS = (pd.Timestamp, datetime.datetime, datetime.date,
                     np.datetime64)
_TIMEDELTA_SCALARS = (pd.Timedelta, datetime.timedelta, np.timedelta64)


def _is_datetime_like(x):
    if isinstance(x, (pd.DatetimeIndex,) + _DATETIME_SCALARS):
        return True
    dtype = getattr(x, 'dtype', None)
    return (isinstance(x, (pd.Series, np.ndarray)) or _is_dataarray(x)) and (
        dtype.kind == 'M' or isinstance(dtype, pd.DatetimeTZDtype))


def _datetime_to_number(x, kind):
    """Convert datetime-like ``x`` to 'unix' seconds or 'doy' (both UTC),
    keeping its shape. pandas and NumPy only: the boundary is on the CPU."""
    if isinstance(x, pd.DatetimeIndex):
        idx, shape = x, None
    elif isinstance(x, pd.Series):
        idx, shape = pd.DatetimeIndex(x), None
    elif isinstance(x, np.ndarray):
        idx, shape = pd.DatetimeIndex(x.ravel()), x.shape
    else:
        idx, shape = pd.DatetimeIndex([pd.Timestamp(x)]), ()
    if idx.tz is not None:
        idx = idx.tz_convert('UTC').tz_localize(None)
    if kind == 'unix':
        out = ((idx - pd.Timestamp('1970-01-01')) / pd.Timedelta('1s'))
    elif kind == 'doy':
        out = idx.day_of_year
    else:
        raise ValueError(f'unknown time representation {kind!r}')
    # a writable copy: pandas returns read-only views, which numba rejects
    out = np.array(out)
    if shape is None:
        return out
    return out.reshape(shape)[()] if shape == () else out.reshape(shape)


def _timedelta_to_seconds(x):
    if isinstance(x, _TIMEDELTA_SCALARS):
        return pd.Timedelta(x).total_seconds()
    return x


def _times_follow_data(arguments, times):
    """
    Move the ``times`` arguments, converted from datetimes to NumPy arrays on
    the CPU, to the library and device of the other array arguments, so
    datetimes work with data from any library and don't decide where the
    calculation runs.
    """
    others = [v for k, v in arguments.items() if k not in times
              and is_array_api_obj(v) and not isinstance(v, np.generic)]
    xp = array_namespace(*others)
    if xp is np_compat:
        return
    device = array_api_compat.device(max(others, key=_size))
    for k in times:
        if isinstance(arguments[k], np.ndarray):
            arguments[k] = xp.asarray(arguments[k], device=device)


# --- time inside the core ---------------------------------------------------

def unix_to_civil(t, xp):
    """
    Calendar fields of ``t``, float64 seconds since 1970-01-01 UTC.

    Returns ``year, month, day, day_of_year`` as float arrays. Uses only
    array API operations, so it runs on any library and inside ``jax.jit``.
    The algorithm is ``civil_from_days`` of H. Hinnant,
    https://howardhinnant.github.io/date_algorithms.html, which is exact for
    the proleptic Gregorian calendar.
    """
    floor = xp.floor
    days = floor(t / 86400.)
    z = days + 719468.
    era = floor(z / 146097.)
    doe = z - era * 146097.                                 # [0, 146096]
    yoe = floor((doe - floor(doe / 1460.) + floor(doe / 36524.)
                 - floor(doe / 146096.)) / 365.)            # [0, 399]
    doy_march = doe - (365. * yoe + floor(yoe / 4.) - floor(yoe / 100.))
    mp = floor((5. * doy_march + 2.) / 153.)                # March is 0
    day = doy_march - floor((153. * mp + 2.) / 5.) + 1.
    month = xp.where(mp < 10., mp + 3., mp - 9.)
    year = yoe + era * 400. + xp.astype(month <= 2., t.dtype)
    leap = ((((year % 4.) == 0.) & ((year % 100.) != 0.))
            | ((year % 400.) == 0.))
    day_of_year = xp.where(doy_march >= 306., doy_march - 305.,
                           doy_march + 60. + xp.astype(leap, t.dtype))
    return year, month, day, day_of_year


def rolling_sum(x, t, window, xp):
    """
    Sum of ``x`` over the time window ``(t - window, t]`` at each ``t``, like
    pandas ``rolling(window, closed='right').sum()``.

    ``t`` is 1-D, increasing, in seconds; ``window`` is in seconds. Works for
    irregular timestamps and has no data-dependent shapes, so it also runs
    inside ``jax.jit``.

    Sums are differences of a running total, so their absolute error grows
    with the total (about 1e-16 times it), where pandas' error grows with the
    window sum. Neither is exact, so a comparison with a threshold that a
    sum can equal (quantized rainfall, say) needs a tolerance to decide ties
    the same way on every library.
    """
    n = t.shape[0]
    total = xp.cumulative_sum(x, axis=-1, include_initial=True)
    start = xp.searchsorted(t, t - window, side='right')
    end = xp.arange(1, n + 1, device=array_api_compat.device(t))
    return (xp.take(total, end, axis=-1)
            - xp.take(total, start, axis=-1))


def take_last(values, flags, xp):
    """
    At each position, ``values`` at the most recent position (inclusive)
    where ``flags`` is True, or 0 before the first. A forward fill without a
    loop or data-dependent shapes; ``flags`` is 1-D.
    """
    count = xp.cumulative_sum(xp.astype(flags, xp.int64))
    last = xp.searchsorted(count, count, side='left')
    return xp.where(count > 0, xp.take(values, last, axis=-1), 0.)


def where_safe(cond, x, f, fill_value, xp, safe=1.):
    """
    ``f(x)`` where ``cond`` is True, else ``fill_value``.

    ``f`` sees ``safe`` in place of ``x`` where ``cond`` is False, so values
    that ``cond`` excludes (a zero divisor, a zero base with a fractional
    power) can make neither the result nor its derivative inf or NaN. This
    "double where" is what autodiff needs; ``array_api_extra.apply_where``
    evaluates ``f`` everywhere under ``jax.jit``.
    """
    return xp.where(cond, f(xp.where(cond, x, safe)), fill_value)


def erf(x, xp):
    """The error function, which the standard does not include."""
    if array_api_compat.is_torch_namespace(xp):
        import torch
        return torch.special.erf(x)
    if array_api_compat.is_jax_namespace(xp):
        import jax.scipy.special
        return jax.scipy.special.erf(x)
    if array_api_compat.is_cupy_namespace(xp):
        import cupyx.scipy.special
        return cupyx.scipy.special.erf(x)
    import scipy.special
    if array_api_compat.is_numpy_namespace(xp) or \
            array_api_compat.is_dask_namespace(xp):
        return scipy.special.erf(x)  # a ufunc, so Dask stays lazy
    # other libraries: round trip through NumPy with DLPack
    out = scipy.special.erf(np.from_dlpack(x))
    return xp.from_dlpack(out)


# --- labeled containers -----------------------------------------------------

def _unwrap(x, found, masks):
    if isinstance(x, pd.Series):
        found.append(x)
        return x.to_numpy()
    if isinstance(x, pd.DataFrame):
        found.append(x)
        return {k: v.to_numpy() for k, v in x.items()}
    if isinstance(x, pd.DatetimeIndex):
        found.append(x)
    if isinstance(x, np.ma.MaskedArray):
        masks.append(np.ma.getmaskarray(x))
        return np.ma.filled(x.astype(np.result_type(x.dtype, np.float64)),
                            np.nan)
    return x


def _wrap_masked(out, masks):
    """Mask outputs wherever any masked input was masked, as ufuncs do."""
    if isinstance(out, dict):
        return {k: _wrap_masked(v, masks) for k, v in out.items()}
    if isinstance(out, tuple):
        return tuple(_wrap_masked(o, masks) for o in out)
    if isinstance(out, (np.ndarray, np.generic)):
        shape = np.shape(out)
        mask = np.zeros(shape, dtype=bool)
        for m in masks:
            mask = mask | np.broadcast_to(m, shape)
        return np.ma.masked_array(out, mask=mask)
    return out


def _is_numpy(x):
    return not is_array_api_obj(x) or isinstance(x, (np.ndarray, np.generic))


def _wrap_pandas(out, index, name):
    # pandas holds NumPy data. A result from another library, for data
    # given with a DatetimeIndex, stays in that library
    if isinstance(out, dict):
        if all(_is_numpy(v) for v in out.values()):
            return pd.DataFrame(out, index=index)
        return out
    if isinstance(out, tuple):
        return tuple(_wrap_pandas(o, index, None) for o in out)
    if isinstance(out, np.ndarray) and out.ndim == 1:
        return pd.Series(out, index=index, name=name)
    return out


def _call_xarray(func, arguments, call, name):
    """
    Align DataArray arguments, give each one the union of all dimensions
    (size 1 where it has none), call ``func`` on the underlying arrays and
    relabel the result.

    Size-1 dimensions let NumPy-style broadcasting do the rest, so a term
    that depends only on time is computed once per time, not once per grid
    cell.
    """
    import xarray as xr

    keys = [k for k, v in arguments.items() if _is_dataarray(v)]
    # join as xarray arithmetic does (inner by default)
    aligned = xr.align(*(arguments[k] for k in keys),
                       join=xr.get_options()['arithmetic_join'])
    dims, sizes = [], {}
    for da in aligned:
        dims += [d for d in da.dims if d not in dims]
        sizes.update(da.sizes)
    full = tuple(sizes[d] for d in dims)
    # unlabeled arrays broadcast by position, as in xarray arithmetic, and
    # may not add dimensions, which would leave the result unlabeled
    for k, v in arguments.items():
        if k in keys or not (is_array_api_obj(v) or isinstance(v, list)):
            continue
        shape = tuple(np.shape(v) if isinstance(v, list) else v.shape)
        try:
            ok = np.broadcast_shapes(shape, full) == full
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(
                f'{k} has shape {shape}, which does not broadcast to '
                f'{full}, the shape of dimensions {tuple(dims)} of the '
                'DataArray arguments. Pass a DataArray to broadcast by '
                'dimension name.')
    # private names: a parameter may share its name with a coordinate
    coords = xr.Dataset({f'_{i}': da for i, da in enumerate(aligned)}).coords
    for k, da in zip(keys, aligned):
        data = da.transpose(*[d for d in dims if d in da.dims]).data
        shape = tuple(sizes[d] if d in da.dims else 1 for d in dims)
        arguments[k] = array_namespace(data).reshape(data, shape)
    out = call()

    if name is _INFER:
        names = {da.name for da in aligned}
        name = names.pop() if len(names) == 1 else None

    def relabel(o, name):
        if is_array_api_obj(o) and getattr(o, 'ndim', 0) == len(dims):
            o = array_namespace(o).broadcast_to(o, full)
            return xr.DataArray(o, coords=coords, dims=dims, name=name)
        return o

    if isinstance(out, dict):
        return xr.Dataset({k: relabel(v, k) for k, v in out.items()})
    if isinstance(out, tuple):
        return tuple(relabel(o, None) for o in out)
    return relabel(out, name)


def _all_scalar(args):
    return all(x is None or isinstance(x, (_SCALAR_TYPES, str))
               for x in args)


def _scalarize(out):
    # mimic NumPy, which returns NumPy scalars rather than 0-d arrays
    if isinstance(out, np.ndarray) and out.ndim == 0:
        return out[()]
    if isinstance(out, dict):
        return {k: _scalarize(v) for k, v in out.items()}
    if isinstance(out, tuple):
        return tuple(_scalarize(o) for o in out)
    return out


_INFER = object()


def _convert_arrays(arguments, names):
    """
    Find the namespace of the ``names`` arguments, convert them in place with
    :py:func:`asarrays`, and return the namespace. The values of a dict
    argument are converted too.
    """
    values = [arguments.get(k) for k in names]
    if not any(isinstance(v, dict) for v in values):  # the common case
        xp = array_namespace(*values)
        for k, x in zip(names, asarrays(*values, xp=xp)):
            if k in arguments:
                arguments[k] = x
        return xp
    keys, values = [], []
    for k in names:
        v = arguments.get(k)
        if isinstance(v, dict):
            keys += [(k, key) for key in v]
            values += list(v.values())
        else:
            keys.append((k, None))
            values.append(v)
    xp = array_namespace(*values)
    dicts = {}
    for (k, key), x in zip(keys, asarrays(*values, xp=xp)):
        if key is not None:
            dicts.setdefault(k, {})[key] = x
        elif k in arguments:
            arguments[k] = x
    arguments.update(dicts)
    return xp


def labeled_io(name=_INFER, arrays=(), time=None, timedelta=(),
               index_time=None):
    """
    Make a function written against the array API (the core) into a pvlib
    function (the boundary) that accepts and returns pandas objects, xarray
    DataArrays, NumPy masked arrays, scalars and arrays of any library.

    The core takes the array namespace as a keyword-only argument ``xp``, and
    gets the parameters named in ``arrays`` as arrays of that namespace, with
    a common floating-point dtype and device. Once per call, the boundary

    1. converts pandas Series to NumPy arrays and DataFrames to dicts of
       them, datetimes to numbers (see ``time``) in the library and on the
       device of the other array arguments, and masked arrays to arrays
       with NaN at masked positions. xarray DataArrays are aligned and
       broadcast by dimension name and replaced by their underlying arrays,
       which may be from any array library;
    2. finds the namespace of the ``arrays`` arguments and converts them with
       :py:func:`asarrays`;
    3. calls the core, and wraps its result: a 1-D NumPy array as a Series
       and a dict of them as a DataFrame on the index of the first pandas
       argument, an array as a DataArray and a dict as a Dataset, masked
       wherever any masked argument was, or as a NumPy scalar if all
       arguments were scalars.

    pvlib functions call each other's cores, ``f.core(..., xp=xp)``, with
    arguments that are already arrays of ``xp``, so this work isn't repeated.
    The decorated function's signature leaves out ``xp``.

    Parameters
    ----------
    name : str, None, or callable, optional
        Name of a Series or DataArray result. By default it is the name shared
        by all the labeled arguments, or None if they differ, as in pandas
        arithmetic. A callable gets the bound arguments and returns the name,
        or ``labeled_io.INFER`` for the default.
    arrays : tuple of str
        Parameters that are arrays, or None, or dicts of arrays. Other
        parameters, such as model names and flags, reach the core unchanged.
    time : dict, optional
        Maps parameter names to the numeric time the function works with,
        ``'unix'`` or ``'doy'``. Datetime-like arguments are converted;
        numeric ones pass through unchanged. A DatetimeIndex argument also
        indexes a pandas result.
    timedelta : tuple of str, optional
        Parameters converted from Timedelta-like to float seconds.
    index_time : str, optional
        A parameter that, when None, is filled with the DatetimeIndex of the
        first Series argument. For functions whose pandas API takes time from
        an index. It should also appear in ``time``. A ValueError is raised
        if it is None and there is no DatetimeIndex to take it from.
    """
    time = time or {}

    def decorator(func):
        sig = inspect.signature(func)
        xp_param = sig.parameters.get('xp')
        params = [p for p in sig.parameters.values() if p.name != 'xp']
        if (xp_param is None or xp_param.kind is not xp_param.KEYWORD_ONLY
                or any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
                       for p in params)):
            raise TypeError(f'{func.__name__} must take a keyword-only xp, '
                            'and no *args or **kwargs')
        names = [p.name for p in params]
        positional = [p.name for p in params
                      if p.kind is p.POSITIONAL_OR_KEYWORD]
        defaults = {p.name: p.default for p in params
                    if p.default is not p.empty}

        def bind(args, kwargs):
            # like sig.bind and apply_defaults, but faster. A missing
            # argument is left for the call to the core to report
            if len(args) > len(positional):
                raise TypeError(f'{func.__name__}() takes {len(positional)} '
                                f'positional arguments but {len(args)} were '
                                'given')
            given = dict(zip(positional, args))
            for k, v in kwargs.items():
                if k in given:
                    raise TypeError(f'{func.__name__}() got multiple values '
                                    f'for argument {k!r}')
                if k not in sig.parameters or k == 'xp':
                    raise TypeError(f'{func.__name__}() got an unexpected '
                                    f'keyword argument {k!r}')
                given[k] = v
            return given

        def call_core(arguments, times):
            if times:
                _times_follow_data(arguments, times)
            xp = _convert_arrays(arguments, arrays)
            return func(**arguments, xp=xp)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            given = bind(args, kwargs)
            arguments = {**defaults, **given}
            # defaults are plain values, so only given arguments need
            # unwrapping
            keys = list(given)
            out_name = name(arguments) if callable(name) else name
            if index_time and arguments.get(index_time) is None:
                _fill_index_time(arguments, index_time)
                if index_time not in given:
                    keys.append(index_time)
            for k in timedelta:
                if k in arguments:
                    arguments[k] = _timedelta_to_seconds(arguments[k])
            if any(_is_dataarray(arguments[k]) for k in keys):
                import xarray as xr
                # signature order sets the order of the result's dimensions
                arguments = {k: arguments[k] for k in names if k in arguments}
                times = []
                for k, kind in time.items():
                    v = arguments.get(k)
                    if _is_dataarray(v) and _is_datetime_like(v):
                        arguments[k] = xr.DataArray(
                            _datetime_to_number(v.values, kind),
                            dims=v.dims, coords=v.coords, name=v.name)
                        times.append(k)
                return _call_xarray(func, arguments,
                                    lambda: call_core(arguments, times),
                                    out_name)
            found, masks, times = [], [], []
            for k in keys:
                v = arguments[k]
                # convert time first: unwrapping a tz-aware Series of
                # datetimes would give an object array of Timestamps
                if k in time and _is_datetime_like(v):
                    if isinstance(v, (pd.Series, pd.DatetimeIndex)):
                        found.append(v)
                    arguments[k] = _datetime_to_number(v, time[k])
                    times.append(k)
                else:
                    arguments[k] = _unwrap(v, found, masks)
            scalar = not found and not masks and _all_scalar(
                [arguments[k] for k in keys])
            out = call_core(arguments, times)
            if not found:
                if masks:
                    return _wrap_masked(out, masks)
                return _scalarize(out) if scalar else out
            if out_name is _INFER:
                names_ = {x.name for x in found if isinstance(x, pd.Series)}
                out_name = names_.pop() if len(names_) == 1 else None
            index = found[0] if isinstance(found[0], pd.Index) \
                else found[0].index
            return _wrap_pandas(out, index, out_name)

        wrapper.__signature__ = sig.replace(parameters=params)
        wrapper.core = func
        return wrapper
    return decorator


labeled_io.INFER = _INFER


def _fill_index_time(arguments, index_time):
    series = [v for v in arguments.values() if isinstance(v, pd.Series)]
    if series and isinstance(series[0].index, pd.DatetimeIndex):
        arguments[index_time] = series[0].index
    else:
        raise ValueError(f'{index_time} is required unless the data are '
                         'Series with a DatetimeIndex')
