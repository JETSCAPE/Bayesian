"""Author-only export from trusted local paper caches; never fit an emulator.

The historical author loader reads trusted local model pickles. Collaborators
use only the resulting numerical checkpoints through review_state.py, not this
workspace-dependent exporter. Both original sources and caches remain read-only.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from bayesian.emulation.review_state import load_prediction_state, save_prediction_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=['inclusive', 'jets', 'all'], default=['all'])
    args = parser.parse_args()
    root = args.workspace.resolve()/'paper_local_checks_20261019'
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('_review_trusted_registry', root/'queue_models.py')
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    args.output.mkdir(parents=True, exist_ok=True)
    records = {}
    for case in args.cases:
        print('LOAD_FROZEN_MODEL', case, flush=True)
        bank = api.load_bank(case, experimental=False)
        model = bank['mean_model']
        original_manifest = root/'queue/MANIFEST.json'
        provenance = dict(case=case, model_kind=api.spec(case)['model_kind'],
            fit_id=next(iter(bank['models'].values()))['joint_group_fit_id'],
            source_manifest_sha256=hashlib.sha256(original_manifest.read_bytes()).hexdigest(),
            raw_feature_layout='Own-subset training only; no output-mask surrogate')
        path = args.output/(case+'.npz')
        if not path.exists():
            save_prediction_state(path, model, features=bank['features'],
                                  design_ids=bank['raw']['Design_indices'], provenance=provenance)
        restored, rows, ids, meta = load_prediction_state(path)
        if meta['provenance'] != provenance:
            raise ValueError('Existing checkpoint has a different source identity')
        np.testing.assert_array_equal(rows, np.asarray(bank['features'], dtype=str))
        np.testing.assert_array_equal(ids, bank['raw']['Design_indices'])
        points = np.vstack([bank['raw']['Design_validation'], [[.2, .4], [.5, .5], [.75, .5]]])
        expected, actual = model.predict(points), restored.predict(points)
        differences = {}
        for key in expected:
            # Saving strided arrays can change BLAS contraction order by roundoff.
            np.testing.assert_allclose(actual[key], expected[key], rtol=5e-12, atol=5e-13)
            differences[key] = float(np.max(np.abs(expected[key]-actual[key])))
        np.savez_compressed(args.output/(case+'_reference.npz'), points_unit=points, **expected)
        records[case] = dict(provenance, prediction_state_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             field_max_absolute_difference=differences, points=len(points))
        print('PREDICTION_ROUNDTRIP', case, path.stat().st_size, differences, flush=True)
        del bank, model, restored
        gc.collect()
    (args.output/'EXPORT.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
