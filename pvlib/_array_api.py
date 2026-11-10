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

Coordinates
-----------
Some functions work along one dimension of their data: time series functions
step through time, and spectral functions integrate over wavelength. They
declare the parameter that holds the coordinate of that dimension
(``labeled_io``'s ``coordinate``), ``times`` or ``wavelength``, and take it
from one place per input type: the labels of pandas objects, the dimension
of a DataArray of the coordinate, or a 1-D array along the last axis of
other arrays. The core always gets the dimension on the last axis.
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
# the argument types of the boundary's fast path, by exact type
_FAST_TYPES = frozenset({bool, int, float, np.float64, str, type(None)})
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


def asarrays(*xs, xp, unix=()):
    """
    Convert ``xs`` to floating-point arrays with a common dtype and device.

    The dtype is the one NumPy arithmetic on the floating-point inputs gives
    (arrays and NumPy scalars; Python floats don't change it), or float64 if
    there are none. The device is that of the arrays, see
    :py:func:`common_device`. Masked values of NumPy masked arrays become
    NaN, so their fill values never enter a calculation
    (:py:func:`labeled_io` restores the mask). ``None`` passes through.

    The positions in ``unix`` hold Unix seconds. They become float64 and
    don't take part in choosing the others' dtype, since float32 resolves
    them only to 128 s; floating-point ones that aren't float64 raise
    ``TypeError``. Datetimes and strings raise ``TypeError`` everywhere.
    """
    for i in unix:
        _check_unix(xs[i], xp)
    if xp is np_compat:
        floats = [x for i, x in enumerate(xs) if i not in unix
                  and isinstance(x, (np.ndarray, np.generic))
                  and x.dtype.kind == 'f']
        dtype = np.result_type(*floats) if floats else np.float64
        return tuple(None if x is None else
                     _np_asarray(x, np.float64 if i in unix else dtype)
                     for i, x in enumerate(xs))
    arrays = [(i, x) for i, x in enumerate(xs) if is_array_api_obj(x)
              and not isinstance(x, np.generic)]
    floats = [x for i, x in arrays
              if i not in unix and xp.isdtype(x.dtype, 'real floating')]
    dtype = xp.result_type(*floats) if floats else xp.float64
    device = common_device([x for _, x in arrays])
    return tuple(None if x is None else
                 xp.asarray(x, dtype=xp.float64 if i in unix else dtype,
                            device=device)
                 for i, x in enumerate(xs))


def _check_unix(x, xp):
    if (is_array_api_obj(x) and xp.isdtype(x.dtype, 'real floating')
            and x.dtype != xp.float64):
        raise TypeError(f'Unix seconds must be float64, not {x.dtype}: '
                        'float32 resolves them only to 128 s')


def common_device(arrays):
    """
    The device of ``arrays``, or None if there are none.

    Arrays on different devices raise ``ValueError``, as the array API
    standard recommends, since moving data between devices is slow and easy
    to miss. 0-d arrays are the exception: they move to the device of the
    others, so a tilt given as ``torch.tensor(30.)`` on the CPU works with
    data on a GPU.
    """
    devices = []
    for x in arrays:
        # None for arrays traced by jax.jit
        device = array_api_compat.device(x) if x.ndim > 0 else None
        if device is not None and device not in devices:
            devices.append(device)
    if len(devices) > 1:
        raise ValueError(
            'arrays are on different devices: '
            f'{", ".join(str(d) for d in devices)}. Move them to one '
            'device; only 0-d arrays are moved for you.')
    if devices:
        return devices[0]
    return array_api_compat.device(arrays[0]) if arrays else None


def _np_asarray(x, dtype):
    if isinstance(x, np.ma.MaskedArray):
        return _filled(x, dtype)
    if isinstance(x, str):
        raise TypeError(f'expected numbers, got the string {x!r}')
    # NumPy would read datetimes as nanoseconds and numeric strings as
    # numbers
    x = np.asarray(x)
    if x.dtype.kind in 'mM':
        raise TypeError(f'expected numbers, got {x.dtype} values. Times '
                        'since 1970 are pvlib.tools.datetime_to_unix(times)')
    if x.dtype.kind in 'SU':
        raise TypeError(f'expected numbers, got strings ({x.dtype})')
    return x.astype(dtype, copy=False)


def _filled(x, dtype=None):
    """The masked array ``x`` as an ndarray of ``dtype``, NaN where it is
    masked. The dtype defaults to that of ``x`` if it is floating point,
    else float64."""
    if dtype is None:
        dtype = x.dtype if x.dtype.kind == 'f' else np.float64
    return np.ma.filled(x.astype(dtype), np.nan)


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


def _time_to_number(x, kind):
    """:py:func:`_datetime_to_number`, for DataArrays too, keeping their
    labels."""
    if _is_dataarray(x):
        import xarray as xr
        return xr.DataArray(_datetime_to_number(x.values, kind), dims=x.dims,
                            coords=x.coords, name=x.name)
    return _datetime_to_number(x, kind)


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
    device = common_device(others)
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

    ``t`` is 1-D, increasing, in seconds, and is the last axis of ``x``;
    ``window`` is in seconds. Works for irregular timestamps and has no
    data-dependent shapes, so it also runs inside ``jax.jit``.

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
    At each position along the last axis, ``values`` at the most recent
    position (inclusive) where ``flags`` is True, or 0 before the first. A
    forward fill without a loop or data-dependent shapes. ``values`` and
    ``flags`` broadcast together; the result has their shape.

    The standard's ``searchsorted`` is 1-D, so the running counts of flags
    in each row are offset by the row number times ``n + 1``, where ``n`` is
    the length of the last axis. That makes all rows one increasing 1-D
    array, which ``searchsorted`` searches at once.
    """
    values, flags = xp.broadcast_arrays(values, flags)
    shape = tuple(flags.shape)
    n = shape[-1]
    rows = math.prod(shape[:-1])
    count = xp.cumulative_sum(xp.astype(flags, xp.int64), axis=-1)
    offset = xp.arange(rows, dtype=xp.int64,
                       device=array_api_compat.device(count)) * (n + 1)
    key = xp.reshape(count + xp.reshape(offset, shape[:-1] + (1,)), (-1,))
    last = xp.searchsorted(key, key, side='left')
    found = xp.reshape(xp.take(xp.reshape(values, (-1,)), last), shape)
    return xp.where(count > 0, found, 0.)


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


def trapezoid(y, x, xp):
    """
    Integral of ``y`` over ``x`` along the last axis by the trapezoidal
    rule, like ``scipy.integrate.trapezoid(y, x=x, axis=-1)``. ``x`` is 1-D
    or broadcasts with ``y``. The standard has no integration functions.
    """
    return xp.sum(xp.diff(x) * (y[..., 1:] + y[..., :-1]) / 2., axis=-1)


def scipy_array_api():
    """
    True if SciPy's array API support is on, so that ``scipy.special``
    calls the functions of the array's library.

    SciPy reads the environment variable ``SCIPY_ARRAY_API`` once, when it is
    first imported, so this asks SciPy rather than the environment. It uses
    a private SciPy name; if that moves, the answer is False and pvlib
    dispatches special functions itself.
    """
    try:
        from scipy._lib import _array_api
    except ImportError:
        return False
    return bool(getattr(_array_api, 'SCIPY_ARRAY_API', False))


def erf(x, xp):
    """
    The error function, which the standard does not include.

    ``scipy.special.erf`` when :py:func:`scipy_array_api` is True; otherwise
    the error function of PyTorch, JAX or CuPy for their arrays, and SciPy's
    for others.
    """
    if scipy_array_api():
        import scipy.special
        return scipy.special.erf(x)
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

def _unwrap(x, masks):
    if isinstance(x, pd.Series):
        return x.to_numpy()
    if isinstance(x, pd.DataFrame):
        return {k: v.to_numpy() for k, v in x.items()}
    if isinstance(x, np.ma.MaskedArray):
        masks.append(np.ma.getmaskarray(x))
        return _filled(x)
    return x


def _common_index(found):
    """
    The index of the pandas objects ``found``, (name, object) pairs.

    They are unwrapped to arrays, which are matched by position, so their
    indexes must be equal: pvlib doesn't align them as pandas arithmetic
    does, which would also make NaN where they differ. Times in different
    time zones are equal if they are the same instants, as for pandas; the
    result has the index, and so the time zone, of the first.
    """
    (first, x), *others = found
    index = _index_of(x)
    for name, x in others:
        other = _index_of(x)
        if other is not index and not _same_index(other, index):
            raise ValueError(
                f'{first} and {name} have different indexes. pvlib does not '
                'align pandas inputs; make their indexes equal first, for '
                'example with reindex')
    return index


def _index_of(x):
    return x if isinstance(x, pd.Index) else x.index


def _same_index(a, b):
    if a.equals(b):
        return True
    # Index.equals is False for the same instants in two time zones. Naive
    # and localized times are never equal: pandas doesn't compare them
    return (isinstance(a, pd.DatetimeIndex) and isinstance(b, pd.DatetimeIndex)
            and a.tz is not None and b.tz is not None and len(a) == len(b)
            and bool((a == b).all()))


def _broadcasts_to(shape, target):
    try:
        return np.broadcast_shapes(shape, target) == target
    except ValueError:
        return False


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
            # an output that doesn't depend on a masked input may have
            # fewer dimensions than it, poa_ground_diffuse for example
            if _broadcasts_to(m.shape, shape):
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


def _call_xarray(arguments, call, name, time_dim=None, reduce=False):
    """
    Call ``call`` on the underlying arrays of the DataArray ``arguments``
    with :py:func:`xarray.apply_ufunc`, and label the result.

    ``apply_ufunc`` aligns the DataArrays as xarray arithmetic does, and
    gives each one the dimensions of all of them, with size 1 where it has
    none, instead of broadcasting them. So a term that depends only on time
    is computed once per time, not once per grid cell. Dask arrays pass
    through unchanged.

    ``time_dim`` names the time dimension of a time series function. It is
    the last axis of every array the core gets, and of the result, wherever
    it was in the inputs; arrays without it get a size-1 last axis. With
    ``reduce``, the core's results lack that axis.
    """
    import xarray as xr

    keys = [k for k, v in arguments.items() if _is_dataarray(v)]
    dims = []
    for k in keys:
        dims += [d for d in arguments[k].dims if d not in dims]
    if time_dim is None:
        core = [[]] * len(keys)
    else:
        core = [[time_dim] if time_dim in arguments[k].dims else []
                for k in keys]
        dims = [d for d in dims if d != time_dim] + [time_dim]
    called = {}

    def call_on_data(*data):
        if time_dim is not None:
            # apply_ufunc moves the time dimension last, and gives arrays
            # without it no axis for it
            data = [x if c else array_namespace(x).expand_dims(x, axis=-1)
                    for x, c in zip(data, core)]
        # apply_ufunc leaves out leading size-1 dimensions, so the arrays
        # broadcast to the shape of all the dimensions
        full = np.broadcast_shapes(*(tuple(x.shape) for x in data))
        _check_unlabeled(arguments, keys, full, dims)
        arguments.update(zip(keys, data))
        out = call()
        if isinstance(out, dict):
            outputs = list(out.values())
        elif isinstance(out, tuple):
            outputs = list(out)
        else:
            outputs = [out]
        # apply_ufunc labels one output; the others get its labels below.
        # Outputs with fewer dimensions are broadcast and copied, since
        # broadcast arrays are read-only views
        shape = full[:-1] if reduce else full
        outputs = [o if tuple(o.shape) == shape
                   else _broadcast_copy(o, shape) for o in outputs]
        called.update(out=out, outputs=outputs)
        return outputs[0]

    # join as xarray arithmetic does (inner by default). The results are
    # other quantities, so the inputs' attributes (units, ...) are dropped
    first = xr.apply_ufunc(
        call_on_data, *(arguments[k] for k in keys), dask='allowed',
        input_core_dims=core,
        output_core_dims=[[]] if time_dim is None or reduce
        else [[time_dim]],
        join=xr.get_options()['arithmetic_join'], keep_attrs=False)
    out = called['out']
    labeled = [first] + [first.copy(data=o) for o in called['outputs'][1:]]
    if isinstance(out, dict):
        names = list(out)
    elif isinstance(out, tuple):
        names = [None] * len(out)
    else:
        # apply_ufunc names the result as xarray arithmetic does
        names = [first.name if name is _INFER else name]
    for da, n in zip(labeled, names):
        da.name = n
    if isinstance(out, dict):
        return xr.Dataset(dict(zip(names, labeled)))
    if isinstance(out, tuple):
        return tuple(labeled)
    return first


def _broadcast_copy(x, shape):
    xp = array_namespace(x)
    return xp.asarray(xp.broadcast_to(x, shape), copy=True)


def _check_unlabeled(arguments, keys, full, dims):
    # unlabeled arrays broadcast by position, as in xarray arithmetic, and
    # may not add dimensions, which would leave the result unlabeled
    for k, v in arguments.items():
        if k in keys or not (is_array_api_obj(v) or isinstance(v, list)):
            continue
        shape = tuple(np.shape(v) if isinstance(v, list) else v.shape)
        if not _broadcasts_to(shape, full):
            raise ValueError(
                f'{k} has shape {shape}, which does not broadcast to '
                f'{full}, the shape of dimensions {tuple(dims)} of the '
                'DataArray arguments. Pass a DataArray to broadcast by '
                'dimension name.')


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


def _convert_arrays(arguments, names, unix=()):
    """
    Find the namespace of the ``names`` arguments, convert them in place with
    :py:func:`asarrays`, and return the namespace. The values of a dict
    argument are converted too. ``unix`` names the arguments in Unix
    seconds.
    """
    values = [arguments.get(k) for k in names]
    if not any(isinstance(v, dict) for v in values):  # the common case
        xp = array_namespace(*values)
        positions = [i for i, k in enumerate(names) if k in unix]
        for k, x in zip(names, asarrays(*values, xp=xp, unix=positions)):
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
    positions = [i for i, (k, key) in enumerate(keys)
                 if key is None and k in unix]
    dicts = {}
    for (k, key), x in zip(keys, asarrays(*values, xp=xp, unix=positions)):
        if key is not None:
            dicts.setdefault(k, {})[key] = x
        elif k in arguments:
            arguments[k] = x
    arguments.update(dicts)
    return xp


def labeled_io(name=_INFER, arrays=(), time=None, timedelta=(),
               coordinate=None, reduce=False):
    """
    Make a function written against the array API (the core) into a pvlib
    function (the boundary) that accepts and returns pandas objects, xarray
    DataArrays, NumPy masked arrays, scalars and arrays of any library.

    The core takes the array namespace as a keyword-only argument ``xp``, and
    gets the parameters named in ``arrays`` as arrays of that namespace, with
    a common floating-point dtype and device. Once per call, the boundary

    1. converts datetimes to numbers (see ``time``) in the library and on
       the device of the other array arguments, pandas Series to NumPy
       arrays and DataFrames to dicts of them, and masked arrays to arrays
       with NaN at masked positions. pandas arguments must have equal
       indexes, since the arrays are matched by position; ``ValueError``
       otherwise. xarray DataArrays are aligned and broadcast by dimension
       name with :py:func:`xarray.apply_ufunc` and replaced by their
       underlying arrays, which may be from any array library;
    2. finds the namespace of the ``arrays`` arguments and converts them with
       :py:func:`asarrays`;
    3. calls the core, and wraps its result: a 1-D NumPy array as a Series
       and a dict of them as a DataFrame on the index of the first pandas
       argument, an array as a DataArray and a dict as a Dataset, masked
       wherever any masked argument was, or as a NumPy scalar if all
       arguments were scalars.

    Calls whose arguments are all numbers, strings or None skip steps 1 and
    2: the core gets the numbers as NumPy float64 scalars, which NumPy
    handles faster than 0-d arrays, unless the function has ``timedelta``
    or ``coordinate`` parameters.

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
        Parameters that are arrays, or None, or dicts of arrays. Datetimes and
        strings given for them raise ``TypeError``. Other parameters, such as
        model names and flags, reach the core unchanged.
    time : dict, optional
        Maps parameter names to the numeric time the function works with,
        ``'unix'`` or ``'doy'``. Datetime-like arguments are converted;
        numeric ones pass through unchanged. ``'unix'`` arguments are always
        float64, see :py:func:`asarrays`. A DatetimeIndex argument also
        indexes a pandas result.
    timedelta : tuple of str, optional
        Parameters converted from Timedelta-like to float seconds.
    coordinate : str, optional
        The parameter that holds the coordinate of the dimension a function
        works along: ``times`` of a time series function, ``wavelength`` of
        a spectral one. It defaults to None and comes last in the
        signature; a time is also listed in ``time``. The coordinate then
        comes from one place, by the type of the other ``arrays`` arguments:

        - pandas objects: their labels. For a time, the DatetimeIndex of
          the first Series. For other coordinates, the index of Series and
          the columns of DataFrames, which are then matrices whose rows are
          labeled by their index, such as spectra by time; these labels
          must be equal. If the parameter is given too, it must equal the
          labels (times compared as instants) or ``ValueError`` is raised.
        - DataArrays: the parameter, which must be a 1-D DataArray. Its
          dimension is the one, wherever it is in the other DataArrays.
        - anything else: the parameter, which must be 1-D. The dimension is
          the last axis of the other arrays.

        If one of them is missing, or a DataArray is given without a
        DataArray of the coordinate, ``ValueError`` is raised. In all three
        cases the core gets the dimension on the last axis of its arrays.
    reduce : bool, default False
        Whether the core removes the ``coordinate`` dimension, as an
        integral over wavelength does. The result then has the other
        dimensions: a scalar for a Series, and a Series on the index of a
        DataFrame. Otherwise the result has the dimension on its last axis.
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

        unix = [k for k, kind in time.items() if kind == 'unix']

        def call_core(arguments, times):
            if times:
                _times_follow_data(arguments, times)
            xp = _convert_arrays(arguments, arrays, unix)
            return func(**arguments, xp=xp)

        # functions with Timedelta defaults or a coordinate need the full
        # boundary even for scalars
        fast = not (timedelta or coordinate)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            given = bind(args, kwargs)
            arguments = {**defaults, **given}
            if fast and all(type(v) in _FAST_TYPES for v in given.values()):
                # numbers, strings and None need no unwrapping, time
                # conversion or namespace lookup. NumPy scalars are float64
                # 0-d arrays, as asarrays would make them, but faster
                for k in arrays:
                    v = arguments.get(k)
                    if type(v) is str:  # np.float64 would read '30' as 30.
                        raise TypeError(f'expected numbers for {k}, got the '
                                        f'string {v!r}')
                    if v is not None:
                        arguments[k] = np.float64(v)
                return _scalarize(func(**arguments, xp=np_compat))
            # defaults are plain values, so only given arguments need
            # unwrapping
            keys = list(given)
            out_name = name(arguments) if callable(name) else name
            time_dim = None
            if coordinate:
                time_dim = _coordinate(arguments, coordinate, arrays,
                                       coordinate in time)
                if coordinate not in given:
                    keys.append(coordinate)
            for k in timedelta:
                if k in arguments:
                    arguments[k] = _timedelta_to_seconds(arguments[k])
            times = [k for k in keys
                     if k in time and _is_datetime_like(arguments[k])]
            if any(_is_dataarray(arguments[k]) for k in keys):
                # times that aren't DataArrays become unlabeled arrays,
                # which broadcast by position
                for k in times:
                    arguments[k] = _time_to_number(arguments[k], time[k])
                # signature order sets the order of the result's dimensions
                arguments = {k: arguments[k] for k in names if k in arguments}
                return _call_xarray(arguments,
                                    lambda: call_core(arguments, times),
                                    out_name, time_dim, reduce)
            if coordinate and coordinate not in time:
                last, lead = _unwrap_matrices(arguments, keys)
                if last is not None:
                    out = call_core(arguments, times)
                    return _wrap_matrix(
                        out, last, lead, reduce,
                        None if out_name is _INFER else out_name)
            found, masks = [], []
            for k in keys:
                v = arguments[k]
                if isinstance(v, (pd.Series, pd.DataFrame, pd.DatetimeIndex)):
                    found.append((k, v))
                # convert time first: unwrapping a tz-aware Series of
                # datetimes would give an object array of Timestamps
                arguments[k] = (_time_to_number(v, time[k]) if k in times
                                else _unwrap(v, masks))
            if found:
                index = _common_index(found)
            scalar = not found and not masks and _all_scalar(
                [arguments[k] for k in keys])
            out = call_core(arguments, times)
            if not found:
                if masks:
                    return _wrap_masked(out, masks)
                return _scalarize(out) if scalar or reduce else out
            if out_name is _INFER:
                names_ = {x.name for _, x in found if isinstance(x, pd.Series)}
                out_name = names_.pop() if len(names_) == 1 else None
            return _wrap_pandas(out, index, out_name)

        wrapper.__signature__ = sig.replace(parameters=params)
        wrapper.core = func
        return wrapper
    return decorator


labeled_io.INFER = _INFER


def _coordinate(arguments, param, arrays, is_time):
    """
    Apply the rule of ``labeled_io``'s ``coordinate`` to ``param``: fill it
    in from the labels of pandas data, check it, and return the name of its
    dimension in DataArray data, or None.
    """
    t = arguments.get(param)
    data = {k: arguments[k] for k in arrays
            if k != param and arguments.get(k) is not None}
    if _is_dataarray(t) or any(_is_dataarray(v) for v in data.values()):
        if not (_is_dataarray(t) and t.ndim == 1):
            raise ValueError(
                f'{param} must be a 1-D DataArray, such as da.{param}, when '
                'the data are DataArrays: its dimension says which one it '
                'is in the data')
        return t.dims[0]
    if is_time:
        labels = next((v.index for v in data.values()
                       if isinstance(v, pd.Series)
                       and isinstance(v.index, pd.DatetimeIndex)), None)
        if labels is not None:
            if t is not None:
                _check_same_times(t, labels, param)
            arguments[param] = labels
            return None
        source = 'Series with a DatetimeIndex'
    else:
        labels = next((_last_labels(v) for v in data.values()
                       if isinstance(v, (pd.Series, pd.DataFrame))), None)
        if labels is not None:
            if t is not None:
                _check_same_labels(t, labels, param)
            t = arguments[param] = labels.to_numpy()
        source = 'Series or DataFrames labeled with it'
    if t is None:
        raise ValueError(f'{param} is required unless the data are {source}')
    if np.ndim(t) != 1:
        raise ValueError(f'{param} must be 1-D, not {np.ndim(t)}-D: it is '
                         'along the last axis of the data')
    n = np.shape(t)[0]
    for k, v in data.items():
        shape = tuple(np.shape(v)) if not isinstance(v, dict) else ()
        if shape and shape[-1] not in (n, 1):
            raise ValueError(
                f'{k} has shape {shape}, but {param} has {n} values. '
                f'{param} is along the last axis of the data; give values '
                'that are constant along it a last axis of size 1')
    return None


def _last_labels(x):
    """The labels of the last axis of a Series or DataFrame."""
    return x.index if isinstance(x, pd.Series) else x.columns


def _check_same_labels(t, labels, param):
    try:
        same = np.array_equal(np.asarray(t, dtype=np.float64),
                              np.asarray(labels, dtype=np.float64))
    except (TypeError, ValueError):
        same = False
    if not same:
        raise ValueError(
            f'{param} differs from the labels of the pandas data. For '
            f'pandas data the labels are the {param}: leave out {param}, or '
            'make it equal the labels')


def _unwrap_matrices(arguments, keys):
    """
    Replace the pandas arguments of a function along a coordinate that isn't
    time by their values. Series are indexed by the coordinate; DataFrames
    have it in their columns and are matrices, whose index labels the rows
    of the result. Returns the labels of the last axis and of the rows,
    each None if no argument has them. Each must be equal in all arguments.
    """
    last = lead = None
    for k in keys:
        v = arguments[k]
        if not isinstance(v, (pd.Series, pd.DataFrame)):
            continue
        labels = _last_labels(v)
        if last is None:
            last, first = labels, k
        elif not _same_index(labels, last):
            raise ValueError(
                f'{first} and {k} have different labels along their last '
                'axis. pvlib does not align pandas inputs; make the labels '
                'equal first, for example by interpolating')
        if isinstance(v, pd.DataFrame):
            if lead is None:
                lead, first_rows = v.index, k
            elif not _same_index(v.index, lead):
                raise ValueError(
                    f'{first_rows} and {k} have different indexes. pvlib '
                    'does not align pandas inputs; make their indexes equal '
                    'first, for example with reindex')
        arguments[k] = v.to_numpy()
    return last, lead


def _wrap_matrix(out, last, lead, reduce, name):
    """Label the result of a function along a coordinate that isn't time:
    ``last`` along its last axis unless the core removed it, ``lead`` along
    its rows."""
    if not isinstance(out, np.ndarray):
        return out
    if reduce:
        if out.ndim == 0:
            return out[()]
        if out.ndim == 1 and lead is not None:
            return pd.Series(out, index=lead, name=name)
    elif out.ndim == 1:
        return pd.Series(out, index=last, name=name)
    elif out.ndim == 2 and lead is not None:
        return pd.DataFrame(out, index=lead, columns=last)
    return out


def _check_same_times(t, index, param):
    """Raise unless ``t`` is the times of ``index``, as instants."""
    try:
        if _is_datetime_like(t):
            given = _datetime_to_number(t, 'unix')
        elif isinstance(t, np.ma.MaskedArray):
            given = _filled(t, np.float64)
        else:
            given = np.asarray(t, dtype=np.float64)
    except (TypeError, ValueError, RuntimeError) as e:
        # the index replaces t, so it must not pass unchecked
        raise TypeError(f'{param} must be datetimes, or Unix seconds that '
                        'NumPy can read, with Series data') from e
    expected = _datetime_to_number(index, 'unix')
    # 1 µs, a few float64 steps of Unix seconds, for seconds calculated
    # otherwise than by pvlib.tools.datetime_to_unix
    if given.shape != expected.shape or not np.allclose(
            given, expected, rtol=0, atol=1e-6):
        raise ValueError(
            f'{param} differs from the index of the Series data. For Series, '
            f'the index is the time: leave out {param}, or make it equal '
            'the index')
