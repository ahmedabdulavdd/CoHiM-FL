#!/usr/bin/env python3
"""Verify tabular completeness and archived Mean±SD; no training or data modifications."""
from pathlib import Path
import pandas as pd
import numpy as np
import json
ROOT=Path(__file__).resolve().parents[1]/"results"/"CoHiM-FL_Split_Seeds"
SEEDS=[7,21,42,84,126]
parts=[]
for seed in SEEDS:
 p=ROOT/f"seed_{seed}"/"main_results"/f"final_test_seed_{seed}.csv"
 df=pd.read_csv(p)
 assert df.shape[0]==20 and set(df.Seed)=={seed},f"Bad main data: {p}"
 assert df.groupby(["Method","Clients"]).size().eq(1).all()
 audit=json.loads((ROOT/f"seed_{seed}"/"data_integrity"/f"data_split_leakage_audit_seed_{seed}.json").read_text())
 assert audit["overall_status"]=="PASS"
 parts.append(df)
data=pd.concat(parts,ignore_index=True)
summary=pd.read_csv(ROOT/"combined_5_seeds"/"02_main_mean_sd_numeric.csv")
for _,r in summary.iterrows():
 group=data.loc[(data.Method==r.Method)&(data.Clients==r.Clients)]
 assert len(group)==5
 for metric in ["BLEU1","BLEU4","METEOR","ROUGE_L","Total_Comm_Cost_MB"]:
  assert np.isclose(group[metric].mean(),r[metric+"_Mean"],atol=1e-9)
  assert np.isclose(group[metric].std(ddof=1),r[metric+"_SD"],atol=1e-9)
print("PASS: 5 seeds × 5 methods × 4 client counts, integrity audit, and main Mean ± SD")
