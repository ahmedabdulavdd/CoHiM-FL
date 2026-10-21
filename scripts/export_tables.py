#!/usr/bin/env python3
"""Produce a fresh, separate five-seed descriptive summary from raw numeric records.
This script never modifies or replaces the historical source CSVs.
"""
from pathlib import Path
import argparse
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SEEDS=[7,21,42,84,126]
SRC=ROOT/'results'/'CoHiM-FL_Split_Seeds'
METRICS=['BLEU1','BLEU4','ROUGE_L','METEOR','Total_Comm_Cost_MB']
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(ROOT/'generated_tables'))
    args=parser.parse_args()
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    frames=[]
    for seed in SEEDS:
        p=SRC/f'seed_{seed}'/'main_results'/f'final_test_seed_{seed}.csv'
        df=pd.read_csv(p)
        if len(df)!=20 or set(df.Seed)!={seed}:raise ValueError(f'Incomplete seed {seed}')
        frames.append(df)
    raw=pd.concat(frames,ignore_index=True)
    counts=raw.groupby(['Method','Clients'])['Seed'].nunique()
    if len(counts)!=20 or not counts.eq(5).all():raise ValueError('Missing groups or seeds')
    agg=raw.groupby(['Method','Clients'],sort=True)[METRICS].agg(['mean','std'])
    agg.columns=[f'{metric}_{kind.upper()}' for metric,kind in agg.columns]
    agg=agg.reset_index()
    agg.insert(2,'N_Seeds',5)
    agg.to_csv(dest/'main_mean_sample_sd_historical_labels.csv',index=False)
    display=agg.copy()
    display['Method']=display['Method'].replace({'Proposed_CBA_NHFL_MI_Q1':'Proposed_CoHiM_FL'})
    display.to_csv(dest/'main_mean_sample_sd_publication_labels.csv',index=False)
    print('Wrote independently calculated summary and label-only display copy to',dest)
    print('Raw values preserved; display copy is NOT a model re-run.')
if __name__=='__main__':main()
