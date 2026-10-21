#!/usr/bin/env python3
"""Deterministically reconstruct study-level test missing-view masks from the
published Colab algorithm and archived ordered 668-study test manifest.

This is a DERIVED artifact, not an originally saved mask file. The function
copies the logic and signed torch int64 calculations from the source notebook.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_SPLIT_SEED = 42
MISSING_RATES = [0., .3, .6, .9]
PROTOCOLS = ('lateral', 'frontal', 'alternate')


def hash_uniform(index_tensor, salt=0, seed=DATA_SPLIT_SEED):
    x = index_tensor.to(torch.int64) + int(seed) * 1000003 + int(salt) * 9176
    x = (x ^ (x >> 16)) * 2246822519
    x = (x ^ (x >> 13)) * 3266489917
    x = x ^ (x >> 16)
    return torch.remainder(x.abs(), 1000000).float() / 1000000.0


def masks(frame):
    ordered = frame.loc[frame.partition == 'Test', 'study_uid'].astype(str).tolist()
    n = len(ordered)
    if n != 668 or len(set(ordered)) != n:
        raise ValueError('Unexpected held-out test partition')
    universe = torch.arange(n, dtype=torch.long)
    scores = hash_uniform(universe, salt=73).cpu().numpy()
    indices = np.arange(n, dtype=np.int64)
    order = np.lexsort((indices, scores))
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n, dtype=np.int64)
    frontal = hash_uniform(universe, salt=91).cpu().numpy() < 0.5
    records = []
    for rate in MISSING_RATES:
        count = int(round(rate * n))
        is_masked = ranks < count
        for protocol in (('none',) if rate == 0 else PROTOCOLS):
            for i, uid in enumerate(ordered):
                masked_view = ('none' if not is_masked[i] else
                               ('frontal' if (protocol == 'alternate' and frontal[i]) else
                                'lateral' if protocol == 'alternate' else protocol))
                records.append(dict(
                    sample_index=i, study_uid=uid, split='Test', data_split_seed=DATA_SPLIT_SEED,
                    missing_rate_requested=rate, missing_protocol=protocol,
                    masked=int(is_masked[i]), masked_view=masked_view,
                    provenance='reconstructed_from_notebook_deterministic_algorithm'
                ))
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(ROOT/'data/derived_missing_view_masks.csv'))
    args = parser.parse_args()
    split = pd.read_csv(ROOT/'data/split_manifest_3337.csv',dtype={'study_uid':'string'})
    out = masks(split)
    summary = out.groupby(['missing_rate_requested','missing_protocol'],sort=True)['masked'].agg(['sum','count'])
    assert len(out) == 6680
    assert set(summary['count']) == {668}
    assert set(summary['sum']) == {0,200,401,601}, summary.to_string()
    # Verify every seed's original numeric result agrees with the independently computed count.
    for seed in [7,21,42,84,126]:
        source=pd.read_csv(ROOT/f'results/CoHiM-FL_Split_Seeds/seed_{seed}/missing_modality_results/01_raw_missing_modality_all_protocols.csv')
        for (r,protocol), grp in source.groupby(['Missing_Rate','Missing_Protocol']):
            target=int(summary.loc[(r,protocol),'sum'])
            if not (grp['Masked_Samples'].astype(int) == target).all():
                raise AssertionError(f'Missing-count mismatch: seed {seed}, rate {r}, protocol {protocol}')
    dest=Path(args.output)
    dest.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(dest,index=False)
    print(f'PASS 5-seed count audit; derived {len(out)} test-study×condition mask rows: {dest}')
    print('PROVENANCE: derived using documented deterministic formula and archived ordered test UIDs.')
    print('These are NOT originally saved per-study mask files.')

if __name__ == '__main__':main()
