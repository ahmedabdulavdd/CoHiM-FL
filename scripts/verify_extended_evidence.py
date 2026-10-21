#!/usr/bin/env python3
"""Audit extra source archives published with the research repository (no model training).

Run with python scripts/verify_extended_evidence.py
Checks are comparisons of archived data, not model-retraining evidence.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SEEDS=(7,21,42,84,126)
BASE=ROOT/'results/CoHiM-FL_Split_Seeds'

def main():
    src=pd.read_csv(ROOT/'data/original_data_split_membership_seed_42.csv')
    published=pd.read_csv(ROOT/'data/split_manifest_3337.csv')
    assert src.uid.is_unique and len(src)==3337
    assert src.Partition.value_counts().to_dict()=={'Train':2335,'Validation':334,'Test':668}
    s=src[['Partition','uid']].rename(columns={'Partition':'partition','uid':'study_uid'}).assign(study_uid=lambda d:d.study_uid.astype(str)).sort_values('study_uid').reset_index(drop=True)
    p=published[['partition','study_uid']].assign(study_uid=lambda d:d.study_uid.astype(str)).sort_values('study_uid').reset_index(drop=True)
    assert s.equals(p),'Original membership does not match published portable split'
    combined=[]
    metrics=['BLEU1','BLEU4','ROUGE_L','METEOR','Label_F1_micro','Label_F1_weighted','Jaccard_micro','Hamming_Loss']
    for seed in SEEDS:
        sub=BASE/f'seed_{seed}'/'reconstruction_missing_ablation'
        detail=pd.read_csv(sub/f'reconstruction_missing_ablation_seed_{seed}.csv')
        means=pd.read_csv(sub/f'reconstruction_missing_ablation_mean_seed_{seed}.csv')
        paired=pd.read_csv(sub/f'reconstruction_missing_ablation_paired_seed_{seed}.csv')
        assert len(detail)==20 and len(means)==8 and len(paired)==4
        assert set(detail.Seed)=={seed} and sorted(detail.Missing_Rate.unique())==[0.,0.3,0.6,0.9]
        assert set(detail.Ablation_Variant)=={'Full_Proposed','w_o_Missing_Reconstruction'}
        for _,r in means.iterrows():
            subset=detail[(detail.Ablation_Variant==r.Ablation_Variant)&(detail.Missing_Rate==r.Missing_Rate)]
            assert len(subset)>0
            for m in metrics:
                assert np.isclose(subset[m].mean(),r[m],rtol=0,atol=1e-9),(seed,r.Ablation_Variant,r.Missing_Rate,m)
        combined.append(means)
    allmeans=pd.concat(combined,ignore_index=True)
    summary=pd.read_csv(BASE/'combined_5_seeds'/'11_reconstruction_missing_mean_sd_numeric.csv')
    assert len(summary)==8
    for _,r in summary.iterrows():
        group=allmeans[(allmeans.Ablation_Variant==r.Ablation_Variant)&(allmeans.Missing_Rate==r.Missing_Rate)]
        assert set(group.Seed)==set(SEEDS)
        for m in metrics:
            assert np.isclose(group[m].mean(),r[m+'_Mean'],rtol=0,atol=1e-9)
            assert np.isclose(group[m].std(ddof=1),r[m+'_SD'],rtol=0,atol=1e-9)
    for seed in (42,84,126):
        metadata=json.loads((ROOT/f'results/clinical_provenance/seed_{seed}/generated_reports_seed_{seed}_meta.json').read_text())
        assert metadata['n_reports']==13360
        clinical=pd.read_csv(BASE/f'seed_{seed}/generated_report_clinical/generated_report_chexbert_seed_{seed}.csv')
        assert len(clinical)==20 and set(clinical.Seed)=={seed}
    assert len(list((ROOT/'results/diagnostics/seed_21').glob('*.png')))==3
    print('PASS: original 3337-study membership matches portable split')
    print('PASS: 5 seeds × 20 detailed reconstruction rows; 40 per-seed summary rows; 5-seed means and SD')
    print('PASS: CheXbert numeric tables and archived metadata for seeds 42,84,126')
    print('PASS: 3 diagnostic figure files for seed 21')
    print('LIMIT: not a full rerun; generated report text and checkpoints intentionally not included.')
if __name__=='__main__': main()
