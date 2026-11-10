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


.. _array-design-tiers:

Tiers of support, and what each one decided
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The goals fall into five tiers, each a promise of its own to keep:

#. **Arrays of the standard**, on any device, with pandas in and pandas out.
#. **Labeled grids:** xarray in, xarray out.
#. **Lazy and compiled arrays:** Dask without computing, and ``jax.jit``.
#. **Gradients** with PyTorch and JAX.
#. **Dimensions with a coordinate:** time series and spectral models on
   arrays without labels, and on N-D data.

The prototype goes through all five, with 36 functions, to find what each
tier asks of the public API before any of it is released. The aim is that
adding a tier later adds functions and tests, and doesn't change a signature
or a result that users already rely on. This is why the design is larger
than a first tier would need by itself.

.. list-table::
   :header-rows: 1
   :widths: 44 18 38

   * - Decision
     - Tier
     - Seen by
   * - ``xp`` is keyword-only and hidden; the core is ``f.core``
     - 1
     - contributors
   * - array parameters are declared in ``arrays``
     - 1
     - contributors
   * - one floating point dtype and one device per call; libraries don't mix
     - 1
     - users of other libraries
   * - pandas is unwrapped, so indexes must be equal
     - 1
     - pandas users
   * - time is a number in the core, and Unix seconds are float64
     - 1
     - users of other libraries
   * - a core returns a dict, which becomes a DataFrame or a Dataset
     - 1, 2
     - contributors; a few return types changed
   * - DataArrays get size-1 dimensions through ``apply_ufunc``
     - 2
     - nobody: it is how the boundary works
   * - no in-place updates, value-dependent shapes or branches on values;
       value checks go through ``known_any``
     - 3
     - contributors
   * - excluded values are kept out of the calculation, with ``where_safe``
     - 4
     - contributors
   * - an optional coordinate (``times``, ``wavelength``), and its
       dimension on the last axis
     - 5
     - users of time series and spectral functions on arrays

Two things follow from the table.

**The tiers can be adopted one at a time.** Only tiers 1 and 5 put anything
in a public signature or result, and tier 5 only adds an optional parameter
to functions that took Series alone. A first contribution can be a few
tier 1 functions; the ``arrays`` declaration and ``f.core`` it carries are
what let the later tiers arrive without changing it.

**Tiers 3 and 4 are rules for writing cores, not API.** Nothing of them
shows in a signature, so they can be promised function by function. A
function says in its entry in ``tests/_supported_functions.py`` what it
promises, and the tests and the user guide follow that entry. Gradients are
the tier that asks most of contributors, so they are promised by default
and a function can decline, see :ref:`array-design-autodiff`.


.. _array-design-always-on:

Why support is always on
~~~~~~~~~~~~~~~~~~~~~~~~

SciPy and scikit-learn make users turn array API support on. They keep
their NumPy code path next to the new one, so the switch chooses between two
behaviors, and existing users are untouched until they flip it.

pvlib has one copy of each model. The boundary runs for NumPy and pandas
input whether or not anyone passes another library's arrays, so a switch
could only decide whether PyTorch or JAX arrays are accepted, and those
mostly failed before. It would protect nobody, and would be global state for
users to manage. SciPy describes its own switch as temporary.

What a switch gives those projects is room to keep changing things. pvlib
gets that by documenting input from libraries other than NumPy as
experimental. It doesn't cover the changes that reach NumPy and pandas
users, which no switch would hold back; they are listed under
:ref:`array-design-open-questions`.


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

**One floating point dtype.** Arrays reach the core with the dtype NumPy
arithmetic gives the floating point inputs, arrays and NumPy scalars (Python
floats don't change it), or float64 if none is floating point. Unix seconds
are the exception: they stay float64, see :ref:`array-design-time`.
The standard leaves promotion between integer and floating point arrays
unspecified (`type promotion
<https://data-apis.org/array-api/latest/API_specification/type_promotion.html>`_),
so converting first gives the same result on every library. float32 inputs
stay float32, for users who want speed on a GPU.

**One device.** Arrays on different devices raise ``ValueError``, as the
standard recommends (`device support
<https://data-apis.org/array-api/latest/design_topics/device_support.html>`_):
moving data between a GPU and the CPU is slow and easy to miss. 0-d arrays
are the exception. A user may pass the tilt as
``torch.tensor(30., requires_grad=True)``, which is on the CPU, with weather
data on a GPU, and moving one number costs nothing. scikit-learn instead
moves all inputs to the library and device of one argument, ``X`` for
estimators; pvlib functions have no such argument.

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

The cost is that Series with different indexes can't be aligned. With pandas
arithmetic, a mismatch gave NaN. Matched by position, the values of
different times would be paired without an error, so the boundary raises
``ValueError`` unless the indexes are equal. The usual mismatch is data with
gaps, which users can reindex first. Localized times that are the same
instants in different time zones count as equal, as pandas aligns them; the
result keeps the index of the first pandas argument, where pandas would give
UTC. Checking costs nothing when the Series
share an index object, as the columns of a DataFrame do, and about 70 µs for
a year of 1-minute data otherwise.


Why xarray gets size-1 dimensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The boundary calls the core through `xarray.apply_ufunc
<https://docs.xarray.dev/en/stable/generated/xarray.apply_ufunc.html>`_,
which aligns the DataArray arguments as xarray arithmetic does, and gives
each one the dimensions of all of them, with size 1 where it has none,
instead of broadcasting them. The core's ordinary broadcasting does the
rest, so a term that depends only on time is calculated once per time, not
once per grid cell. For 8,760 hours on a 40 × 40 grid,
:py:func:`~pvlib.solarposition.spa_python` takes 1.0 s, against 28 s with
fully broadcast inputs.

``apply_ufunc`` is xarray's own tool for applying a function to the arrays
inside DataArrays, so alignment, coordinates and Dask follow xarray's rules.
pvlib adds two things: every output of a core that returns several is
labeled, and unlabeled arrays, which broadcast by position as in xarray
arithmetic, raise if they would add a dimension, since the result could
then not be labeled.

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
a solar position that is silently minutes off. The boundary converts the
``'unix'`` arguments to float64 separately from the other arrays, so integer
seconds stay exact with float32 data, and raises ``TypeError`` for
floating point seconds that aren't float64.

Calendar fields come from H. Hinnant's `civil_from_days
<https://howardhinnant.github.io/date_algorithms.html>`_, written with
floor division so it runs on any library and inside ``jax.jit``. It matches
pandas exactly on 730,017 timestamps from 1700 to 2300.
``_xa.rolling_sum`` uses a cumulative sum and ``searchsorted`` instead of a
loop, so it handles irregular timestamps and keeps shapes fixed.


.. _array-design-time-series:

Why time series take time from one place
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most models are elementwise: each time step stands alone. Time series
models step through time. :py:func:`~pvlib.soiling.hsu` sums rain over a
window and accumulates dust until it rains, and
:py:func:`~pvlib.snow.coverage_nrel` accumulates the snow that has slid off
since the last snowfall. They need the times, for the length of each step,
and the direction of time in the data, to know what to accumulate along. A
Series carries both in its index. A DataArray has dimension names but no
rule for which one is time, and a NumPy or PyTorch array has neither.
Before the port these functions took only Series.

The design had to meet four requirements:

- **No breaking changes.** pandas calls work as before.
- **Arrays of any library,** plain NumPy and PyTorch included, since
  gradients are much of the point of the port.
- **N-D data,** such as many sites or orientations, in one call.
- **One rule for every time series function,** not one per function.

Four options were considered.

**Labels only.** Time comes from the index of Series and from a dimension
of DataArrays, as in pyet and xclim, which both use the one named
``time``. This leaves out plain arrays, so
NumPy, PyTorch and JAX users couldn't call these functions at all. Finding
the dimension by name is also guesswork: MetPy matches dimension names
against a regular expression (``time``, ``times``, ``xtime``, ...), and lets
users override the match.

**Required times.** Every function takes ``times`` for every input type,
as NumPy and SciPy take numbers and an ``axis``. That is one rule with no
guessing, but every pandas call of these functions would have to change.

**A second function per model** for arrays, next to the Series one. Nothing
breaks, but each model has two names and two docstrings to keep in step,
and users choose by input type.

**An optional times**, as :py:func:`~pvlib.clearsky.detect_clearsky` has
(``times=None``), and :py:func:`~pvlib.scaling.wvm` for its time step
(``dt=None``). Nothing breaks and each model keeps one function. pvlib uses
this one, with the rules below.

pvlib has removed optional ``times`` parameters before, to simplify its
API. This one is different in two ways: only inputs without an index need
it, and these functions didn't take such inputs before; and it can't
disagree with an index.

.. list-table:: Where other libraries get time
   :header-rows: 1
   :widths: 25 75

   * - Library
     - Time comes from
   * - pandas
     - the index
   * - pyet
     - the index of Series, the ``time`` coordinate of DataArrays
   * - xclim
     - the DataArray dimension named ``time``
   * - MetPy
     - optional arguments, filled from DataArray coordinates found by name
   * - statsmodels
     - optional ``dates`` or ``freq``, filled from a pandas index
   * - NumPy, SciPy
     - numbers, and an ``axis`` argument
   * - pvlib
     - the index of Series, the dimension of a ``times`` DataArray, or
       ``times`` along the last axis of other arrays

None of them meets all four requirements: the libraries that read labels
don't take plain arrays, and the ones that take numbers ask for them on
every call.

The rules
---------

**Series: the index,** as before the port. A ``times`` given as well must
be the same instants, within 1 µs, or ``ValueError`` is raised, so there is
never a second answer to what the time is. The usual mistake this catches
is naive local times, which pvlib takes as UTC.

**DataArrays: a DataArray of times.** Its dimension is time, wherever that
dimension is in the data, and ``times=da.time`` is all it takes. DataArray
data with ``times`` missing, or given as an unlabeled array or a
DatetimeIndex, raise ``ValueError``. There is no fallback to a dimension
named ``time``, which would be guessing by name, or to the last position:
the order of dimensions inside the call follows the arguments, not the way
the user laid out the data.

**Anything else: times along the last axis.** ``times`` is required and
1-D. The last axis of each array argument has its length, or length 1 for
values that are constant in time, such as a tilt for each site. Other
lengths raise ``ValueError``, which catches most data laid out with time
first; square data can't be caught.

The boundary moves the time dimension of DataArrays to the last axis too,
with ``input_core_dims`` in :py:func:`xarray.apply_ufunc`, so the core
always gets time last and is written once, with ``axis=-1``. Results have
time last as well.

**Why the last axis.** SciPy's signal and integration functions, which it
has ported to the standard, work along the last axis by default
(``cumulative_trapezoid``, ``savgol_filter``, ``lfilter``), as does the
standard's ``diff``, and :py:func:`xarray.apply_ufunc` puts core dimensions
last. The cost is that a DataFrame's values have time first, so they need a
transpose, and a value for each site needs a last axis of size 1,
``tilt[:, None]``, where with time first it would broadcast as it is. Time
first would suit DataFrame users, but set pvlib apart from SciPy and
xarray.

**No axis parameter.** It would add a parameter to every time series
function for a choice that ``.T`` or ``moveaxis`` already makes, and
DataArray users name the dimension instead. It can be added later without
breaking calls.

**One times for all sites.** ``times`` is 1-D, so the sites of an N-D call
share their timestamps; sites with different timestamps need separate
calls, or data reindexed onto common times. ``_xa.rolling_sum`` searches
one array of times for all sites.

**Forward fills of N-D data.** The standard's ``searchsorted`` is 1-D, and
:py:func:`~pvlib.snow.coverage_nrel` and :py:func:`~pvlib.soiling.hsu` look
up, for each time, the latest snowfall or cleaning. ``_xa.take_last`` adds
row × (n + 1) to each row's running count of events, n being the number of
times, which makes all rows one increasing 1-D array for a single
``searchsorted``. Shapes stay fixed, so it works inside ``jax.jit`` and on
Dask.

**Changes from before the port.** :py:func:`~pvlib.snow.fully_covered_nrel`
used the first snowfall if pandas could infer a frequency from the index,
and now uses it if all time steps are equally long: the same answer for the
sub-daily data the model is for. With two times, where pandas can't infer a
frequency, it raised before.


.. _array-design-coordinates:

Why the time rule is a coordinate rule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Time is not the only dimension a pvlib function works along. The spectral
functions integrate over wavelength, and they were ported to find out
whether the rules for time were rules for time alone:
:py:func:`~pvlib.spectrum.average_photon_energy` and
:py:func:`~pvlib.spectrum.calc_spectral_mismatch_field`.

**What users see holds.** An optional parameter for the coordinate, filled
from the labels of pandas input, required for other arrays, along the last
axis, and named by a DataArray's dimension: ``wavelength`` works exactly as
``times`` does. pvlib had the pattern already, in
:py:func:`~pvlib.spectrum.sr_to_qe` (``wavelength=None``). Spectra in a
DataFrame have wavelength in their columns, which is the last axis, so the
last axis is also what pandas users of these functions have.

**The decorator was too narrow,** and changed: ``index_time`` became
``coordinate``, with ``reduce`` for cores that remove the dimension. It is
private, so no public signature changed. Three things were specific to
time:

- *The result kept the dimension.* An integral removes it: a Series gives a
  number, a DataFrame a Series on its index, and a DataArray loses the
  dimension.
- *A DataFrame was a dict of columns,* as for the components of sky diffuse
  irradiance. Spectra are a matrix: columns are the coordinate and the
  index labels the rows of the result. Which one a DataFrame is depends on
  the function, so the declaration decides: a matrix for a coordinate other
  than time.
- *The labels were a DatetimeIndex.* Any labels are a coordinate.

**Arguments on different grids need labels.**
:py:func:`~pvlib.spectrum.calc_spectral_mismatch_field` takes a spectral
response, spectra and a reference spectrum, each a Series with wavelengths
of its own, and interpolates between them. Arrays have one ``wavelength``
for all three, and nothing is interpolated: three coordinate parameters
would be the alternative. The interpolation stays in the public function,
for pandas input, and the integral it calls twice is the core. So this
function has no ``core`` of its own, like
:py:func:`~pvlib.solarposition.spa_python`.

**Defaults read from files are boundary work.** The default reference
spectrum is read from a file and interpolated to the wavelengths, with
pandas, then moved to the library and device of the data. Inside
``jax.jit`` the wavelengths have no values yet, so ``e_ref`` must be given
there, and the error says so.

**spectrl2 is not ported, and needs an API change first.**
:py:func:`~pvlib.spectrum.spectrl2` creates the wavelength dimension and
puts it first: its results are 122 × N arrays, and a spectral
``ground_albedo`` is 122 × N too. Three things follow.

- It disagrees with the functions that use its results, which take
  wavelength last, so users transpose in between today.
- It takes only 1-D inputs, since a leading axis can't broadcast against
  N-D ones. A grid of sites or orientations needs wavelength last.
- It would be the first function to create a coordinate dimension, which
  the boundary doesn't do yet: the result of a DataArray call needs the new
  dimension and its coordinate.

The change proposed is wavelength on the last axis of its results, N × 122,
for every input type, after a deprecation period. Wavelength last for the
new input types only would give NumPy and PyTorch users different layouts
for the same call. DataArray results aren't affected, since the dimension
has a name. This changes results for current users of a public function, so
it is for discussion upstream before it is built.

**Wavelength in DataFrames has no single convention.**
:py:func:`~pvlib.spectrum.average_photon_energy` and
:py:func:`~pvlib.spectrum.calc_spectral_mismatch_field` take it in the
columns, with a spectrum in each row.
:py:func:`~pvlib.spectrum.get_reference_spectra` returns it in the index,
and :py:func:`~pvlib.spectrum.sr_to_qe` and
:py:func:`~pvlib.spectrum.qe_to_sr` take it there. Both are natural in
pandas, and the boundary can't tell them apart from the labels without
guessing. The rule for ported functions is the one above: the last axis,
which is the columns. The functions that use the index are left as they
are, and aren't ported.

Left open by this port:

- A time series function could take a DataFrame of sites, but there the
  coordinate is the index, the first axis. The boundary would transpose it.
  DataFrames are not accepted by the time series functions today.
- A coordinate given as a NumPy array with data from another library
  raises, as any mix of libraries does, though datetimes follow the data.
  Coordinates are usually NumPy arrays read from a file; letting them
  follow the data too would be consistent with datetimes.
- No function yet works along two such dimensions, time and wavelength,
  say. They would be the last two axes, in an order to be fixed.
- The test tables in ``tests/_supported_functions.py`` expect a result for
  each input value, so functions with ``reduce`` have tests of their own.


.. _array-design-autodiff:

Why gradients and compilation shape the code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Gradients are promised function by function, and by default.** The
standard says nothing about differentiation, so finite gradients are a rule
of pvlib's own, and they change how a core is written, not how it is called.
A function could therefore be supported without them. They are still the
default, for three reasons: gradients are a use raised in :issue:`2515`; a
model chain is only differentiable if every function in it
is; and the work is small when it is done while porting, since it is nearly
always one of the two patterns below. A new entry in
``tests/_supported_functions.py`` is tested for finite gradients with no
further code, and a failure names the argument and the values that caused
it. A function that can't have them, or whose core they would make hard to
read, gives a reason in its entry (``gradients='...'``) and the user guide
lists it. :py:func:`~pvlib.snow.fully_covered_nrel`, whose result is
boolean, is the only one so far.

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


.. _array-design-dispatch:

Why some helpers dispatch by library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard has no special functions. ``scipy.special`` calls the
functions of the array's library only when SciPy's array API support is on,
which takes the environment variable ``SCIPY_ARRAY_API`` set before SciPy is
first imported. So ``_xa.erf`` uses ``scipy.special.erf`` when SciPy's
support is on, and otherwise calls the ``erf`` of PyTorch, JAX or CuPy
itself. SciPy reads the variable only once, so ``_xa.scipy_array_api``
asks SciPy whether its support is on, through a private SciPy name, rather
than reading the environment. pvlib's CI runs the tests both ways.
``_xa.errstate`` exists because ``np.errstate`` is NumPy's alone.

numba compiles NumPy code only. To keep a single copy of the solar position
formulas, ``spa._formulas(xp)`` runs the Python source of the numba
functions with ``np`` mapped to ``xp``. Four functions that loop over
coefficient tables or multiply by booleans have separate array API versions.


.. _array-design-performance:

Performance
~~~~~~~~~~~

The boundary costs a few microseconds per call. On NumPy arrays of 8,760
values that matters only for the cheapest functions, and pandas input is
faster than before. ``array_namespace`` has a fast path for NumPy and pandas
inputs.

Calls with only numbers, which are common in pvlib use, take a fast path
through the boundary: arguments that are all numbers, strings or None need
no unwrapping, time conversion or namespace lookup. The core then gets NumPy
float64 scalars rather than 0-d arrays, which NumPy handles faster. Without
the fast path, :py:func:`~pvlib.irradiance.get_total_irradiance` took 28 µs
with scalars, of which 11 µs was the boundary and 5 µs the 0-d arrays.

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Scalar call
     - before this work
     - full boundary
     - fast path
   * - :py:func:`~pvlib.irradiance.get_total_irradiance`
     - 12 µs
     - 28 µs
     - 14 µs
   * - :py:func:`~pvlib.irradiance.erbs`
     - 7.1 µs
     - 14 µs
     - 7.5 µs
   * - :py:func:`~pvlib.temperature.sapm_cell`
     - 0.4 µs
     - 9.6 µs
     - 2.2 µs
   * - :py:func:`~pvlib.temperature.faiman`
     - 1.0 µs
     - 6.5 µs
     - 1.4 µs

Functions that were a line of arithmetic, such as
:py:func:`~pvlib.pvsystem.pvwatts_dc` at 0.1 µs, cost 1–2 µs more. Code
that calls a function on scalars in a loop should pass arrays instead.


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


.. _array-design-out-of-scope:

Out of scope
~~~~~~~~~~~~

- **The classes.** :py:class:`~pvlib.modelchain.ModelChain`,
  :py:class:`~pvlib.pvsystem.PVSystem` and
  :py:class:`~pvlib.location.Location` stay on pandas. They are built on
  DataFrames and column names, and porting them would mean a different
  results container. A layer for other array libraries would be written on
  top of the ported functions, outside these classes.
- **DataFrame libraries that don't follow the standard,** such as polars.
  The standard is what lets one copy of a model run on many libraries;
  support for a library outside it would be conversion code in the boundary
  that its users can write in a line, with ``to_numpy``.
- **Fitting functions.** ``pvlib.ivtools``, :py:func:`~pvlib.iam.fit` and
  :py:func:`~pvlib.inverter.fit_sandia` are algorithms around SciPy
  optimizers and least squares. With gradients through the forward models,
  users of other libraries can fit with those libraries' optimizers.
- **Reading and writing data.** ``pvlib.iotools`` returns pandas objects.


.. _array-design-open-questions:

Open questions
~~~~~~~~~~~~~~

- **Changes for NumPy and pandas users.** These arrive with the port of a
  function, whatever library its users pass, so each needs a whatsnew entry
  and a decision on whether it needs a deprecation period first:

  - pandas inputs with different indexes raise ``ValueError``; pandas
    arithmetic aligned them and gave NaN where they differed;
  - some results changed type or name, for example
    :py:func:`~pvlib.clearsky.haurwitz`, which took only a Series and now
    returns a dict for other input, and scalar calls, which return NumPy
    scalars throughout;
  - :py:func:`~pvlib.soiling.hsu` counts rain within 1e-9 mm of
    ``cleaning_threshold`` as reaching it, and
    :py:func:`~pvlib.snow.fully_covered_nrel` decides differently whether
    to use the first snowfall, see :ref:`array-design-time-series`.

- **Documenting support.** A decorator like SciPy's ``xp_capabilities``
  could build the list of supported functions and libraries, what each
  promises (``jax.jit``, Dask, gradients), and the test marks, from one
  place. Today the entry in ``tests/_supported_functions.py`` holds the
  gradient promise, and tests keep the user guide in step with it.
- **Special functions.** Drop pvlib's own dispatch once SciPy's support no
  longer needs ``SCIPY_ARRAY_API``.
- **Increasing times.** :py:func:`~pvlib.soiling.hsu` raises for times that
  go back, as pandas' ``rolling`` did; the snow model, as before the port,
  doesn't check. A check in the boundary would make all time series
  functions agree.
- **The other time series functions.** :py:func:`~pvlib.soiling.kimber`
  and :py:func:`~pvlib.temperature.prilliman` take time from the index and
  would get the same ``times`` when ported;
  :py:func:`~pvlib.temperature.fuentes` also needs a scan, which the
  standard lacks. :py:func:`~pvlib.clearsky.detect_clearsky` and
  :py:func:`~pvlib.scaling.wvm` already have optional time parameters, which
  porting would bring under these rules.


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
