.. _array-libraries-design:

Array library design
====================

This page explains why pvlib's support for array libraries works as it does.
To write or port a function, :ref:`contributing-array-libraries` is all you
need. The design started in :issue:`2515` and is experimental; the open
questions are at the end.


Goals
~~~~~

- **One copy of each model.** The same code runs on NumPy and on any
  library that follows the array API standard, so there is one version to
  maintain and review.
- **Little changes for NumPy and pandas users.** pandas in, pandas out, with
  the same values. The exceptions are described below.
- **Labeled grids.** xarray in, xarray out, broadcasting by dimension name,
  so a model runs over a grid of locations or orientations without a loop.
- **Gradients and compilation.** Models can be differentiated with PyTorch or
  JAX, and compiled with ``jax.jit``. In the gallery example
  :ref:`sphx_glr_gallery_shading_plot_horizon_from_power.py`, gradients cut
  a fit from 5,772 model runs to 169.
- **Small overhead** for the single calls that make up most pvlib use.


.. _array-design-standard:

Why the array API standard
~~~~~~~~~~~~~~~~~~~~~~~~~~

A library that wanted to run on more than NumPy used to have two options.
NumPy's dispatch protocols, ``__array_ufunc__`` and ``__array_function__``
(`NEP 18 <https://numpy.org/neps/nep-0018-array-function-protocol.html>`_),
let Dask and CuPy arrays pass through NumPy functions, but PyTorch and JAX
don't implement them, and ``np.asarray`` calls, common in pvlib, convert
everything to NumPy anyway. The other option was a copy of the code for each
library.

The `array API standard <https://data-apis.org/array-api/latest/>`_ defines
one set of functions and array methods that each library provides (see its
`purpose and scope
<https://data-apis.org/array-api/latest/purpose_and_scope.html>`_). NumPy 2
implements it in its main namespace (`NEP 56
<https://numpy.org/neps/nep-0056-array-api-main-namespace.html>`_), and so
does JAX. `array-api-compat <https://data-apis.org/array-api-compat/>`_
fills the gaps for NumPy 1, PyTorch, CuPy and Dask, and finds the namespace
of a set of arrays. SciPy and scikit-learn use it in the same way.

The standard covers the elementwise math, reductions and array creation that
PV models use. It has no special functions, datetimes or scans, and libraries
that compile or defer work, such as JAX and Dask, may not support functions
whose output shape depends on the data. The sections below say how pvlib
handles each.


.. _array-design-core-boundary:

Why a core and a boundary
~~~~~~~~~~~~~~~~~~~~~~~~~

**pandas and xarray aren't arrays of the standard.** A Series carries an
index and a DataArray carries dimension names and coordinates, and neither
is an array that ``xp`` functions accept. Something has to take them apart
before the math and put the labels back after it, and the math can't do
that, since it doesn't see the labels. That is the boundary; what's left is
the core.

**The work is done once per call.** Before the split, every function found
the namespace and converted its own inputs, so one
:py:func:`~pvlib.irradiance.get_total_irradiance` call did it 20 times as it
went through the functions it calls. Now it does it once, and a call with
scalars dropped from 51 µs to 28 µs. SciPy and scikit-learn pass ``xp`` to
their private helpers for the same reason.

**One decorator instead of code in each function.** SciPy calls
``array_namespace`` at the top of each function. pvlib's boundary also
handles pandas, xarray, masked arrays and time, a few hundred lines that
would otherwise be repeated in every function, and repeated differently. A
test of 25 functions with six input types (``tests/test_input_types.py``)
fails 35 of its 150 cases on code from before this work, where each
function handled lists, masked arrays, Series and DataArrays its own way.

**The namespace is hidden.** Users choose a library by the arrays they
pass, so ``xp`` is not an option for them; it is keyword-only in the core,
the public signature leaves it out, and passing it raises ``TypeError``.
SciPy gives public functions an ``xp`` parameter only when they take no
arrays to infer it from.

**The core is an attribute.** ``f.core`` keeps the core under the model's
name, with its docstring, instead of making it a second private function to
keep in step with the public one.


.. _array-design-declared:

Why array parameters are declared
---------------------------------

The boundary can't tell from a value whether it is data or an option: a
model name, a flag that is 0 or 1, None, or a dict of model parameters.
Converting an option to an array would break the code that compares it.
Listing the array parameters in ``arrays`` makes this explicit, and records
which parameters are data.


Why inputs are converted the way they are
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**One floating point dtype.** Arrays reach the core with the promoted
floating point dtype of the inputs, or float64 if none is floating point.
The standard leaves promotion between integer and floating point arrays
unspecified (`type promotion
<https://data-apis.org/array-api/latest/API_specification/type_promotion.html>`_),
so converting first gives the same result on every library. float32 inputs
stay float32, for users who want speed on a GPU.

**The device of the largest array.** A user may pass the tilt as a 0-d CPU
tensor, ``torch.tensor(30., requires_grad=True)``, with weather data on a GPU.
Moving everything to the device of the largest array keeps the data where it
is. The standard recommends raising on arrays from different devices
(`device support
<https://data-apis.org/array-api/latest/design_topics/device_support.html>`_).
scikit-learn moves inputs to the library and device of one argument, ``X``
for estimators; pvlib functions have no such argument, so size decides.

**Libraries don't mix,** except NumPy with Dask, since Dask arrays are NumPy
arrays computed lazily, and xarray mixes them the same way: a grid may hold
some DataArrays in memory and others in Dask. Python scalars mix with
anything, as the standard allows in arithmetic, and so do datetimes (see
:ref:`array-design-time`).

**Scalars in, NumPy scalar out,** as NumPy ufuncs do, and as pvlib did
before.

**Masked arrays** reach the core with NaN where values are masked, and the
result is masked wherever an input was, as for ufuncs. Fill values such as
-9999 therefore never enter the math. SciPy rejects masked arrays when its
array API support is on, and supports `MArray
<https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html#a-note-on-marray-support>`_,
a masked array library that follows the standard, instead.


Why pandas is unwrapped
~~~~~~~~~~~~~~~~~~~~~~~

pandas operations run only on NumPy, so the core can't use them. The
boundary passes the values of a Series to the core and puts the result back
on the index of the first pandas argument. Skipping pandas' work on every
operation also made Series input faster: 375 µs instead of 1,390 µs for
:py:func:`~pvlib.irradiance.get_total_irradiance` on 8,760 values.

The cost is that Series with different indexes are no longer aligned. With
pandas arithmetic, a mismatch gave NaN; now values are matched by position.


Why xarray gets size-1 dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The boundary aligns DataArray arguments as xarray arithmetic does, and gives
each one the union of the dimensions, with size 1 where it has none. The
core's ordinary broadcasting does the rest, so a term that depends only on
time is calculated once per time, not once per grid cell. For 8,760 hours on
a 40 × 40 grid, :py:func:`~pvlib.solarposition.spa_python` takes 1.25 s,
against 30.1 s with fully broadcast inputs.

Unlabeled arrays broadcast by position, as in xarray arithmetic, and raise
if they would add a dimension, since the result could then not be labeled.

DataArrays can hold Dask arrays and other arrays that aren't NumPy (see
xarray's `duck array
<https://docs.xarray.dev/en/stable/user-guide/duckarrays.html>`_ support);
the core gets the array inside.


.. _array-design-time:

Why time is a number
~~~~~~~~~~~~~~~~~~~~

The standard has no datetime dtype, so time reaches the core as a number,
the one the model uses: Unix seconds for solar position, which ``pvlib.spa``
already worked in, the day of year for extraterrestrial radiation, and
seconds for durations. All are UTC. Datetimes come as pandas and NumPy
objects on the CPU, so the boundary converts them with pandas, then moves
the numbers to the library and device of the other array arguments.
Datetimes therefore work with data from any library, and don't decide where
the calculation runs. The result then has the data's type: a Series would
hold only NumPy data.

Unix seconds need float64. Between 2004 and 2038 they lie between 2\
:sup:`30` and 2\ :sup:`31`, where float32 values are 128 s apart. PyTorch and
JAX create float32 arrays by default, so a check that raises is better than
a solar position that is silently minutes off.

Calendar fields come from H. Hinnant's `civil_from_days
<https://howardhinnant.github.io/date_algorithms.html>`_, written with
floor division so it runs on any library and inside ``jax.jit``. It matches
pandas exactly on 730,017 timestamps from 1700 to 2300.
``_xa.rolling_sum`` uses a cumulative sum and ``searchsorted`` instead of a
loop, so it handles irregular timestamps and keeps shapes fixed.


.. _array-design-autodiff:

Why gradients and compilation shape the code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Double where.** Reverse-mode differentiation of ``xp.where(c, f(x), 0)``
multiplies the derivative of ``f`` by zero where ``c`` is False, and
0 × inf is NaN. ``_xa.where_safe`` gives ``f`` a safe value wherever ``c`` is
False, so neither branch produces inf; JAX's FAQ, `Gradients contain NaN
where using where
<https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where>`_,
describes the trick. ``array_api_extra.apply_where`` doesn't prevent this:
under ``jax.jit`` it calls ``f`` on every element.

**Value checks.** Inside ``jax.jit`` arrays are tracers whose values are
unknown, so ``bool()`` raises, and for Dask it would compute the whole
array. ``_xa.known_any`` runs a check only when its result is known. SciPy
skips such checks for JAX altogether, and advises against telling a jitted
call from an eager one, since JAX's public API can't. pvlib tells them apart
with ``jax.core.Tracer`` so that eager JAX users still get the error; if
that stops working, SciPy's approach is the fallback.

**No in-place updates or value-dependent shapes**, because JAX arrays can't
be modified and ``jax.jit`` and Dask need shapes before they see values. See
the standard's `copies, views and mutation
<https://data-apis.org/array-api/latest/design_topics/copies_views_and_mutation.html>`_
and `data-dependent output shapes
<https://data-apis.org/array-api/latest/design_topics/data_dependent_output_shapes.html>`_,
and JAX's `sharp bits
<https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html>`_.


Why some helpers dispatch by library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard has no special functions. ``scipy.special`` dispatches to
PyTorch, JAX and CuPy only when the environment variable ``SCIPY_ARRAY_API``
is set before SciPy is imported, so ``_xa.erf`` calls each library's own
``erf``. ``_xa.errstate`` exists because ``np.errstate`` is NumPy's alone.

numba compiles NumPy code only. To keep a single copy of the solar position
formulas, ``spa._formulas(xp)`` runs the Python source of the numba
functions with ``np`` mapped to ``xp``. Four functions that loop over
coefficient tables or multiply by booleans have separate array API versions.


.. _array-design-performance:

Performance
~~~~~~~~~~~

The boundary costs a few microseconds per call. On NumPy arrays of 8,760
values that matters only for the cheapest functions, and pandas input is
faster than before. Calls on scalars
are slower: 28 µs for :py:func:`~pvlib.irradiance.get_total_irradiance`
against 12 µs before this work. Code that calls a function on scalars in a
loop should pass arrays instead. ``array_namespace`` has a fast path for
NumPy and pandas inputs, and a fast path for scalars in the boundary is
still to do.


Comparison with SciPy and scikit-learn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SciPy's `array API support
<https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html>`__ and
scikit-learn's `array API support
<https://scikit-learn.org/stable/modules/array_api.html>`__ are the largest
adopters, and both are still experimental.

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: 16 28 28 28

   * -
     - pvlib
     - SciPy
     - scikit-learn
   * - Turned on
     - always
     - by ``SCIPY_ARRAY_API=1`` before import
     - by the ``array_api_dispatch`` setting, and ``SCIPY_ARRAY_API=1``
   * - array-api-compat
     - a dependency
     - a git submodule
     - vendored
   * - Namespace
     - found once per call, by ``labeled_io``
     - found at the top of each function
     - found by ``get_namespace``; helpers take ``xp``
   * - Mixed libraries
     - raise, except NumPy with Dask; scalars and datetimes mix with any
       library
     - raise; scalars and lists mix only with NumPy
     - converted to follow ``X`` or ``y_pred``
   * - Masked arrays
     - NaN inside, mask restored
     - rejected; MArray instead
     - not covered in its guide
   * - Support listed
     - in the user guide
     - in each docstring, by ``xp_capabilities``
     - in the user guide
   * - Tests
     - tables of cases, run on each installed library
     - an ``xp`` fixture, marked by ``xp_capabilities``
     - common checks for estimators tagged ``array_api_support``


Open questions
~~~~~~~~~~~~~~

- **xarray.** ``xarray.apply_ufunc`` also gives arguments size-1 dimensions,
  and handles Dask through ``dask='allowed'``. It could replace pvlib's own
  xarray code if each function declared the names of its outputs.
- **Documenting support.** A decorator like SciPy's ``xp_capabilities``
  could build the list of supported functions and libraries, and the test
  marks, from one place.
- **Devices.** Raise on arrays from different devices, as the standard
  recommends, except for 0-d arrays?
- **A switch.** SciPy and scikit-learn make users turn support on. pvlib
  doesn't, since NumPy and pandas users get the same results apart from the
  changes described above. Whether an experimental feature should still be
  opt-in is open.
- **Special functions.** Use ``scipy.special`` once SciPy's support no longer
  needs ``SCIPY_ARRAY_API``.
- **Scalar overhead.** A fast path in the boundary for calls whose
  arguments are all scalars.


References
~~~~~~~~~~

- Python array API standard: https://data-apis.org/array-api/latest/,
  including its `design topics
  <https://data-apis.org/array-api/latest/design_topics/index.html>`_
- array-api-compat: https://data-apis.org/array-api-compat/
- array-api-strict: https://data-apis.org/array-api-strict/
- array-api-extra: https://data-apis.org/array-api-extra/
- SciPy, Support for the array API standard:
  https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html
- scikit-learn, Array API support:
  https://scikit-learn.org/stable/modules/array_api.html
- NumPy NEP 56, Array API standard support in NumPy's main namespace:
  https://numpy.org/neps/nep-0056-array-api-main-namespace.html
- xarray, Working with numpy-like arrays:
  https://docs.xarray.dev/en/stable/user-guide/duckarrays.html
- JAX, The Sharp Bits:
  https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html
- pvlib discussion: :issue:`2515`
