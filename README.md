CoHiM-FL
Communication-Efficient Hierarchical Multimodal Federated Learning for Medical Report Generation with Missing Modalities

This repository contains the implementation notebooks and archived numerical evidence supporting the CoHiM-FL study prepared for PeerJ Computer Science (AI Application). CoHiM-FL investigates a simulated client–edge–cloud federated setting with compressed model communication and missing-view handling for chest X-ray report generation. The study does not constitute a clinical deployment.
Repository contents
Path	Description
notebooks/	Five independent experiment notebooks (seeds 7, 21, 42, 84, 126), a five-seed aggregation notebook, and additional analysis notebooks.
results/CoHiM-FL_Split_Seeds/combined_5_seeds/	Saved numerical summary tables and raw five-seed comparison tables, including main results, ablation analysis and missing-view evaluation.
data/	Study-level split files, an overlap audit and derived missing-view masks.
scripts/	Utilities for checking notebook sources, verifying archived results, reconstructing masks and exporting tables.
requirements.txt	Main Python dependencies. This is not an exact historical package-version freeze.


Release scope: The repository is being populated in stages. The presence of summary CSV files does not by itself mean that all original seed-level output logs or model checkpoints are available. Check the actual repository contents before attempting every verification script.
Dataset and preprocessing
The underlying data come from the Indiana University Chest X-ray collection, originally distributed through the U.S. National Library of Medicine's Open-i service. The experiment notebooks expect the CSV/image organization provided by the Kaggle distribution.

The recorded experimental subset consists of 3,337 studies, split into 2,335 training, 334 validation, and 668 test studies with a fixed data-split seed of 42. The vocabulary was constructed from training reports, and frozen ImageNet-pretrained DenseNet121 features were used for the image inputs. Federated clients were simulated with non-IID data partitioning. Study-level split information is supplied in data/; patient-level disjointness could not be independently established from the available metadata.

The file data/derived_missing_view_masks.csv is a deterministic reconstruction of the missing-view evaluation masks, not the original saved mask artifact. Original chest X-ray images and radiology-report texts are not redistributed. Obtain these separately from an authorized source and follow its reuse terms.
Requirements
- Python/Jupyter and PyTorch; a GPU-enabled Google Colab runtime is recommended for the training notebooks.
- Core Python packages are listed in [`requirements.txt`](requirements.txt).
- The original exact Python, PyTorch, CUDA and GPU allocations were not preserved as a complete version-pinned environment. The current dependency list is a practical starting point, not a verified replica of the historical environment.
Running the notebooks
1. Acquire the Indiana University Chest X-ray dataset and prepare the CSV/image layout expected by the notebooks. The provided Colab code historically uses a Google Drive archive at /content/drive/MyDrive/Indiana.zip; adjust paths only to point to your legitimately acquired local data.
2. Open each notebooks/01_seed_*_CoHiM_FL_FINAL_PUBLICATION_READY.ipynb notebook in Colab. Mount Google Drive, review dataset and output paths, and run cells in sequence.
3. Execute the five experiment seeds 7, 21, 42, 84, and 126 independently, writing outputs to separate seed directories. Keep the common study-level data split fixed at seed 42.
4. After all five experiment outputs exist, use the notebooks/06_COMBINE_5_SEEDS_...ipynb aggregation notebook to calculate means and sample standard deviations. Preserve a backup of any existing combined files before rerunning the aggregation notebook.
5. Compare regenerated outputs with the archived numerical tables in results/. Differences should be reported, not manually corrected to match published values.
Checking archived numerical files
Once the complete results hierarchy required by each script is present, the verification utilities can be used from the repository root. For example:
python -m pip install -r requirements.txt
python scripts/verify_results.py
python scripts/verify_extended_evidence.py
python scripts/check_notebook_sources.py
The scripts examine saved artifacts; passing these checks is not equivalent to independent GPU retraining, independent clinical validation or proof of full end-to-end reproducibility. Additional scripts may have further input requirements; inspect their source before running them.
Provenance and limitations
The original archived experiment results used the internal identifier Proposed_CBA_NHFL_MI_Q1; the manuscript uses the project name CoHiM-FL. A change of display name does not by itself verify that revised notebooks reproduce historical experiments. Numerical metrics were not modified as part of the naming change.

The numerical evaluation uses five independent seeds. Original model checkpoint binaries and an exact frozen environment are not included in the current repository. Some archived missing-view diagnostic flags remain to be investigated. These limitations should be retained when interpreting the evidence.
Citation, license, and contact
When citing the work, use the final article metadata if and when it is published. No article DOI should be inferred before publication. A code license has not yet been selected; reuse permissions are therefore not automatically granted. Data from external providers remain subject to their own license or usage conditions.
