"""
Record what pvlib before the array API port returns for the calls in
``_array_api_cases``, for ``test_array_api_vs_main.py``.

Run it with pvlib from before the port first on the path, for example the
commit this branch started from::

    mkdir /tmp/pvlib-main
    git archive <commit> pvlib | tar -x -C /tmp/pvlib-main
    PYTHONPATH=/tmp/pvlib-main python tests/_make_array_api_golden.py \\
        --source <commit>

The results go to ``tests/data/array_api_main_results.json``.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

import pvlib

TESTS = Path(__file__).resolve().parent
OUTPUT = TESTS / 'data' / 'array_api_main_results.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--source', required=True,
                        help='what pvlib was run, e.g. a commit')
    args = parser.parse_args()
    if hasattr(pvlib.tools, 'datetime_to_unix'):
        sys.exit(f'pvlib at {Path(pvlib.__file__).parent} has the array API '
                 'port; put pvlib from before it first on PYTHONPATH')

    # pvlib is imported already, so the cases use it, not the branch's
    sys.path.append(str(TESTS.parent))
    from tests import _array_api_cases as cases

    try:
        import numba
        numba_version = numba.__version__
    except ImportError:
        numba_version = None
    results = {}
    for name, case in cases.CASES.items():
        if 'numba' in case.requires and numba_version is None:
            sys.exit(f'{name} needs numba')
        results[name] = cases.run(case)
    meta = {'source': args.source, 'numpy': np.__version__,
            'pandas': pd.__version__,
            'scipy': scipy.__version__, 'numba': numba_version}
    OUTPUT.write_text(json.dumps({'meta': meta, 'cases': results}, indent=1,
                                 sort_keys=True) + '\n')
    errors = sum(r['type'] == 'error' for r in results.values())
    print(f'{len(results)} cases, {errors} raised, written to {OUTPUT}')


if __name__ == '__main__':
    main()
