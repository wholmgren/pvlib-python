.. _contributing-array-libraries:

Supporting array libraries
==========================

Some pvlib functions accept arrays from any library that follows the
`Python array API standard <https://data-apis.org/array-api/latest/>`_, such
as PyTorch, JAX, CuPy and Dask, as well as NumPy, pandas and xarray. The
user guide's :ref:`array_libraries` page lists them. This page shows how to
write such a function, or to port an existing one.

The reasons for the rules are on a separate page,
:ref:`array-libraries-design`. You don't need it to follow them; the
"(why)" links lead to the relevant section.

Support for array libraries is experimental. The helpers live in the private
module ``pvlib/_array_api.py``, imported as ``_xa``, and may change.


The pattern
~~~~~~~~~~~

A supported function has two parts:

- the **core**, the model's math, written against an array namespace ``xp``
  instead of NumPy;
- the **boundary**, made by the decorator ``labeled_io``, which accepts
  pandas objects, xarray DataArrays, masked arrays, scalars and arrays from
  any library, calls the core, and returns the type it was given.

Here is :py:func:`pvlib.iam.martin_ruiz` without its docstring:

.. code-block:: python

    from pvlib import _array_api as _xa
    from pvlib.tools import _cosd


    @_xa.labeled_io(name=None, arrays=('aoi', 'a_r'))
    def martin_ruiz(aoi, a_r=0.16, *, xp):
        if _xa.known_any(a_r <= 0, xp):
            raise ValueError("The parameter 'a_r' cannot be zero or negative.")

        with _xa.errstate(xp, invalid='ignore'):
            iam = (1 - xp.exp(-_cosd(aoi, xp) / a_r)) / (1 - xp.exp(-1 / a_r))
            iam = xp.where(xp.abs(aoi) >= 90.0, 0.0, iam)

        return iam

Inside the core, ``aoi`` and ``a_r`` are arrays of the namespace ``xp``, with
the same floating point dtype and on the same device, whatever the caller
passed. ``xp`` is not part of the public function: users choose the library
by the arrays they pass. (:ref:`why <array-design-core-boundary>`)


Porting a function
~~~~~~~~~~~~~~~~~~

#. Add a keyword-only ``xp`` parameter at the end of the signature:
   ``def f(a, b, model='x', *, xp):``. Don't document it.
#. Decorate the function with ``@_xa.labeled_io(arrays=(...))``, naming each
   parameter that holds numbers. See :ref:`array-libraries-decorator`.
#. Replace NumPy calls with ``xp`` calls from the standard, following
   :ref:`array-libraries-core`.
#. Replace pandas operations, such as those on a time index, following
   :ref:`array-libraries-time`.
#. Call other supported functions through their cores:
   ``aoi_projection.core(..., xp=xp)``.
#. Add an entry for the function to ``SUPPORTED`` in
   ``tests/_supported_functions.py``, and benchmarks, see
   :ref:`array-libraries-testing`. The entry also promises finite
   gradients, unless it says why not; see :ref:`array-libraries-gradients`.
#. Add the function to the list of supported functions in
   ``docs/sphinx/source/user_guide/modeling_topics/array_libraries.rst``. If
   only some of its models or methods are supported, say which. A test checks
   that the list names exactly the functions in ``SUPPORTED``.


.. _array-libraries-decorator:

The decorator
~~~~~~~~~~~~~

``arrays``
    The parameters that hold numbers: scalars or arrays, possibly None. They
    reach the core as arrays of ``xp``. A parameter that holds a dict of
    arrays, such as ``poa_sky_diffuse`` in
    :py:func:`~pvlib.irradiance.poa_components`, can be listed too; its values
    are converted. Leave out strings, flags and dicts of model parameters:
    they reach the core unchanged. (:ref:`why <array-design-declared>`)

``name``
    The name of a Series or DataArray result. By default it is the name
    shared by the labeled arguments, as in pandas arithmetic. Give a string
    for a fixed name, such as ``'aoi'``, or ``None`` for no name. Match the
    name the function returned before it was ported.

``time``
    Parameters that take datetimes, mapped to the number the core gets:
    ``'unix'`` for float64 seconds since 1970-01-01 UTC, ``'doy'`` for the
    day of year. For example ``time={'time': 'unix'}`` in
    :py:func:`~pvlib.solarposition.spa_python`.

``timedelta``
    Parameters that take durations, such as a ``pd.Timedelta``. The core gets
    seconds.

``coordinate``
    For functions that work along one dimension of their data: the parameter
    that holds its coordinate. For time series functions, which take time
    from the index of their Series, it is ``times``, as in
    :py:func:`~pvlib.soiling.hsu`; for spectral functions it is
    ``wavelength``, as in :py:func:`~pvlib.spectrum.average_photon_energy`.
    Add it last, as ``times=None``, list a time in ``time`` too, and copy
    its docstring from one of those functions. The boundary takes the
    coordinate from the labels of pandas objects (the index of Series, and
    for coordinates other than time the columns of DataFrames), from the
    dimension of a DataArray of it, or from a 1-D array along the last axis
    of other arrays, and raises if it is missing or disagrees with the
    labels; the core gets the dimension on the last axis.
    See :ref:`array-libraries-time`.
    (:ref:`why <array-design-time-series>`,
    :ref:`and for other dimensions <array-design-coordinates>`)

``reduce``
    ``True`` if the core removes the ``coordinate`` dimension, as an
    integral over wavelength does. Use ``_xa.trapezoid(y, x, xp)`` for the
    integral. The result then has the other dimensions of the data: a
    number for a Series, a Series for a DataFrame.

The core must return an array, a dict of arrays or a tuple of arrays. The
boundary makes a dict into a DataFrame for pandas input and a Dataset for
xarray input; describe it in the docstring as, for example, "dict or
DataFrame".

For calls with only numbers, strings and None, the core gets NumPy float64
scalars instead of 0-d arrays, which is faster
(:ref:`why <array-design-performance>`). The core must work with both, which
it does if it follows the rules below. ``test_input_types.py`` runs scalars
through this path and the other input types through the full boundary.


.. _array-libraries-core:

Writing the core
~~~~~~~~~~~~~~~~

**Use only the standard.** Most NumPy functions have the same name in the
`API specification <https://data-apis.org/array-api/latest/API_specification/index.html>`_
(``xp.exp``, ``xp.where``, ``xp.maximum``, ``xp.clip``). These are the ones
that differ, and the helpers to use where the standard has nothing:

=========================================  ==========================================
NumPy                                      In a core
=========================================  ==========================================
``np.arccos``, ``np.arctan2``, ...         ``xp.acos``, ``xp.atan2``, ...
``np.power``                               ``xp.pow``, or ``**``
``np.concatenate``                         ``xp.concat``
``np.cumsum``                              ``xp.cumulative_sum``
``x.astype(float)``                        ``xp.astype(x, xp.float64)``
``np.radians``, ``np.degrees``             ``_xa.deg2rad(x)``, ``_xa.rad2deg(x)``
``tools.cosd``, ``tools.sind``, ...        ``tools._cosd(x, xp)``, ``tools._sind(x, xp)``, ...
``np.errstate(...)``                       ``_xa.errstate(xp, ...)``
``scipy.special.erf``                      ``_xa.erf(x, xp)``
``scipy.special.lambertw``                 ``ivtools.utils._lambertw_pvlib(x, xp)``
=========================================  ==========================================

For another special function, add a helper like ``_xa.erf``: it uses
``scipy.special`` when SciPy's array API support is on
(``_xa.scipy_array_api()``), and otherwise the function of each library.
(:ref:`why <array-design-dispatch>`)

**Don't modify arrays in place.** JAX arrays can't be modified. Write
``x = xp.where(mask, 0., x)`` instead of ``x[mask] = 0``.

**Keep shapes independent of values.** Boolean indexing such as ``x[x > 0]``
gives a shape that depends on the data, which Dask and ``jax.jit`` can't
handle. Compute everywhere and select with ``xp.where``. Don't multiply by a
boolean array either (``mask * x``); the standard doesn't allow it. In
``xp.where``, at least one of the two values must be an array.

Computing every branch for every element can cost NumPy users time that
masked assignment saved. Where a branch is expensive, skip it when no
element takes it: ``if _xa.maybe_any(mask, xp):`` is True whenever the
values aren't known yet, so ``jax.jit`` and Dask still get both branches.
``pvlib.singlediode._lambertw_i_from_v`` is an example. An array whose
elements take different branches still pays for both, so add an asv
benchmark with such an array, like ``SingleDiodeBranches`` in
``benchmarks/benchmarks/pvsystem.py``.


.. _array-libraries-gradients:

Gradients
~~~~~~~~~

Each supported function promises, by default, that PyTorch and JAX can
differentiate it: its derivatives are finite wherever its arguments are.
Users fit parameters and optimize designs through chains of functions, and
one function with a NaN derivative breaks the chain.
(:ref:`why <array-design-autodiff>`)

You don't write a gradient test. The function's entry in ``SUPPORTED`` runs
it through ``tests/test_array_api_gradients.py``, which fails with the
argument and the values that gave a derivative that isn't finite. Put the
values a branch excludes into the entry, such as a zenith past 90°, zero
power or a NaN airmass, since that is where gradients break.

Nearly every failure is one of two cases, both fixed with
``_xa.where_safe`` as shown under :ref:`array-libraries-core`:

- **a branch that is left out still computes inf or NaN:** a division by
  zero or a fractional power of zero inside ``xp.where``. Compute it from a
  safe value there:
  ``_xa.where_safe(zeta != 0, zeta, lambda z: 0.0059 / z, 0., xp)``;
- **a NaN input is masked after it is used:** the NaN airmass of night,
  multiplied by something and then set to 0. Keep it out of the
  calculation:
  ``_xa.where_safe(~xp.isnan(am), am, lambda am: ..., xp.nan, xp)``.

The values the function returns don't change, so the other tests need no
changes.

If a function can't have gradients, because its result is boolean or
integer, or if the fix would make its core hard to follow, say so in its
entry instead:

.. code-block:: python

    'snow.fully_covered_nrel': Supported(
        {...}, gradients='its result is boolean'),

The gradient tests then skip it, and it must be named in the Gradients
section of the user guide, which a test checks. Prefer the fix: a function
without gradients takes them from every model that calls it.


.. _array-libraries-time:

Time
~~~~

The standard has no datetime type, so the core gets time as numbers: declare
the parameter in ``time`` and the boundary converts datetimes, to arrays of
the same library and on the same device as the other array arguments.
Numbers pass through unchanged. Inside the core, use these instead of
pandas:

=========================================  ==========================================
pandas                                     In a core
=========================================  ==========================================
``times.year``, ``.month``, ``.day``       ``_xa.unix_to_civil(t, xp)``
``times.dayofyear``                        ``_xa.unix_to_civil(t, xp)``, or ``time={'x': 'doy'}``
``x.rolling(window).sum()``                ``_xa.rolling_sum(x, t, window, xp)``
forward fill of values at flagged steps    ``_xa.take_last(values, flags, xp)``
=========================================  ==========================================

Unix seconds must be float64, since float32 resolves them only to 128
seconds. Declare them with ``time={'x': 'unix'}``: the boundary then
converts them to float64 apart from the other arrays and raises
``TypeError`` for float32. (:ref:`why <array-design-time>`)

A time series function, declared with ``coordinate='times'``, gets ``times`` as
1-D Unix seconds and its data with time on the last axis, whatever the
input type. Write the core for N-D data: pass ``axis=-1`` to
``cumulative_sum`` and the other functions that take an axis, index time
with ``x[..., i]``, and broadcast against ``xp.diff(times)``. The helpers
above all work along the last axis.


.. _array-libraries-testing:

Testing
~~~~~~~

Add an entry for the function to ``SUPPORTED`` in
``tests/_supported_functions.py``. It gives the function's array arguments,
with four values each, its other arguments, the name of a Series result,
the parameters that also take datetimes, and those that must be series
along time, never scalars.
Include values that reach the branches the function excludes, such as a
zenith below the horizon or a NaN airmass. The entry runs the function
through:

- ``tests/test_input_types.py``: Python scalars, lists, NumPy arrays, masked
  arrays, named Series and xarray DataArrays, checking the type and values
  of each result;
- ``tests/test_input_types_mixed.py``: calls that mix these types, and
  datetimes, so that every pair of types for every pair of arguments
  appears;
- (a function with ``reduce`` doesn't fit these tables, which expect a
  result for each input value: name it in ``ELSEWHERE`` in the same file
  and test it as ``tests/test_array_api_wavelength.py`` does)
- ``tests/test_array_api.py``: array-api-strict, PyTorch, JAX and Dask,
  checking that the function returns the type it was given and the values
  NumPy gives;
- ``tests/test_array_api_gradients.py``: gradients with PyTorch and JAX,
  which must be finite wherever the arguments are, unless the entry gives a
  reason in ``gradients``, see :ref:`array-libraries-gradients`;
- ``tests/test_array_api_vs_main.py``: the same calls with Python, NumPy and
  pandas inputs must give what pvlib gave before the port. After adding an
  entry, record those results again with ``tests/_make_array_api_golden.py``,
  whose docstring says how.

``tests/test_array_api.py`` also checks that every public function made with
``labeled_io`` has an entry.

Add tests of your own where the function needs them:

- a function that chains others with ``where_safe`` in them: that the
  gradient of the chain is finite and matches finite differences, like
  ``test_torch_clear_sky_chain_through_night``;
- a function with ``known_any``: that the check raises for eager JAX arrays
  and is skipped inside ``jax.jit``, like
  ``test_martin_ruiz_validates_eager_jax_but_not_traced``;
- a function that takes time: tests in ``tests/test_array_api_time.py``;
- a time series function: an entry in ``FUNCTIONS`` in
  ``tests/test_array_api_time_series.py``, which runs it on data for several
  sites and checks each site against Series input.

A CI job, ``array libraries`` in ``.github/workflows/pytest.yml``, installs
these libraries and runs the whole test suite twice: with SciPy's array API
support off, and on (``SCIPY_ARRAY_API=1``). The other CI jobs don't install
them. Locally, libraries that aren't installed are skipped; ``-rs`` lists
the skips.

.. code-block:: bash

    pip install -e ".[test]" array-api-strict torch jax "dask[array]" xarray
    pytest tests/test_array_api*.py tests/test_input_types.py -rs
    SCIPY_ARRAY_API=1 pytest tests/test_array_api*.py tests/test_input_types.py -rs

For benchmarks, add a ``time_`` method to the ``InputTypes`` class in the
module's file in ``benchmarks/benchmarks``, such as ``IAMInputTypes`` in
``iam.py``. It times the function with a scalar, a NumPy array and a Series
of 8,760 values. See :ref:`array-design-performance` for why the scalar case
matters.


Functions that can't be ported yet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Leave a function on NumPy if it

- steps through time with a loop, as :py:func:`~pvlib.temperature.fuentes`
  does: the standard has no scan;
- calls a SciPy optimizer or root finder, as
  :py:func:`~pvlib.pvsystem.singlediode` does for the maximum power point.
  pvlib doesn't write solvers of its own (:ref:`why <array-design-solvers>`).
  Port the closed-form functions around the solver, so that users of other
  libraries can apply their library's solver to them, and raise an error
  for their arrays that says so, as ``singlediode`` does;
- returns datetimes, as the sunrise and sunset functions do;
- needs results whose shape depends on the data;
- creates a dimension, as :py:func:`~pvlib.spectrum.spectrl2` does for
  wavelength: see :ref:`array-design-coordinates` for what it needs first.

The solar position formulas in ``pvlib/spa.py`` are numba functions.
``spa._formulas(xp)`` runs their Python source on any library, with ``np``
names mapped to ``xp``. Edit them as before, using only NumPy names that the
mapping knows; ``test_spa_formula_numpy_names_have_array_api_equivalents``
checks this.


Helper reference
~~~~~~~~~~~~~~~~

.. currentmodule:: pvlib._array_api

.. autofunction:: labeled_io(name=INFER, arrays=(), time=None, timedelta=(), coordinate=None, reduce=False)

.. autofunction:: known_any

.. autofunction:: is_concrete

.. autofunction:: where_safe

.. autofunction:: maybe_any

.. autofunction:: errstate

.. autofunction:: erf

.. autofunction:: scipy_array_api

.. autofunction:: unix_to_civil

.. autofunction:: rolling_sum

.. autofunction:: take_last

.. autofunction:: trapezoid

.. autofunction:: array_namespace

.. autofunction:: asarrays

.. autofunction:: common_device
