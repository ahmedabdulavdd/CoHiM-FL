#!/usr/bin/env python3
"""Check publicly archived evidence without downloading data or training a model."""
from pathlib import Path
import json, hashlib, sys
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SEEDS=(7,21,42,84,126)
SROOT=ROOT/'results'/'CoHiM-FL_Split_Seeds'
def require(condition,message):
    if not condition: raise AssertionError(message)

def main():
    split=pd.read_csv(ROOT/'data/split_manifest_3337.csv',dtype={'study_uid':'string'})
    require(len(split)==3337,'Invalid number of study-level split entries')
    require(split.study_uid.nunique()==3337,'Duplicated study identifiers')
    require(split.partition.value_counts().to_dict()=={'Train':2335,'Validation':334,'Test':668},'Split counts differ')
    fingerprints=set()
    main=[]
    ablations=[]
    for seed in SEEDS:
        base=SROOT/f'seed_{seed}'
        audit=json.loads((base/'data_integrity'/f'data_split_leakage_audit_seed_{seed}.json').read_text())
        require(audit['overall_status']=='PASS',f'Study/image audit not PASS: {seed}')
        require(audit['patient_audit_status']=='NOT_AVAILABLE_IN_SOURCE_METADATA','Unexpected patient audit flag')
        fingerprints.add(audit['split_fingerprint_sha256'])
        x=pd.read_csv(base/'main_results'/f'final_test_seed_{seed}.csv')
        require(len(x)==20 and set(x.Seed)=={seed},f'Main seed {seed} malformed')
        require(len(x[['Method','Clients']].drop_duplicates())==20, f'Main groups not unique for seed {seed}')
        main.append(x)
        a=pd.read_csv(base/'ablation_results'/f'ablation_final_seed_{seed}.csv')
        require(len(a)==7,f'Ablation missing seed {seed}')
        ablations.append(a)
        m=pd.read_csv(base/'missing_modality_results'/'01_raw_missing_modality_all_protocols.csv')
        require(len(m)==200,f'Missing-modality result count seed {seed} differs')
        require(set(np.round(m.Missing_Rate.unique(),2))=={0.,0.3,0.6,0.9},f'Missingness ratios differ seed {seed}')
        print(f'PASS seed {seed}: main={len(x)} ablation={len(a)} missing={len(m)}')
    require(len(fingerprints)==1,'Different study split across seeds')
    main=pd.concat(main,ignore_index=True)
    summary=pd.read_csv(SROOT/'combined_5_seeds'/'02_main_mean_sd_numeric.csv')
    require(len(main)==100,'Main raw row count')
    metric_cols=('BLEU1','BLEU4','ROUGE_L','METEOR','Total_Comm_Cost_MB')
    for _,r in summary.iterrows():
        g=main[(main.Method==r.Method)&(main.Clients==r.Clients)]
        require(set(g.Seed)==set(SEEDS),f'Seed count {r.Method}/{r.Clients}')
        for c in metric_cols:
            require(np.isclose(g[c].mean(),r[f'{c}_Mean'],rtol=0,atol=1e-9),f'Mean differs {r.Method}/{r.Clients}/{c}')
            require(np.isclose(g[c].std(ddof=1),r[f'{c}_SD'],rtol=0,atol=1e-9),f'SD differs {r.Method}/{r.Clients}/{c}')
    require(len(summary)==20,'Expected 20 aggregated main groups')
    sens=pd.read_csv(ROOT/'results/weight_sensitivity/summary/weight_sensitivity_raw_25_runs.csv')
    require(len(sens)==25,'Expected 25 sensitivity experiments')
    print('PASS fixed 3337-study split; archived means/SD; 5 seeds; 25 sensitivity runs')
    print('NOTE: does not evaluate models, checkpoints, clinical safety, or patient-level split independence.')

if __name__=='__main__': main()
