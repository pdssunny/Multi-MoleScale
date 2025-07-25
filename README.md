# Multi-MoleScale: A Multi-Scale Framework for Molecular Property Prediction

Welcome to the official repository for **Multi-MoleScale**, a novel multi-scale framework that integrates Graph Contrastive Learning (GCL) with sequence-based models (e.g., BERT) for accurate molecular property prediction. This repository provides the code, data, and instructions to reproduce the results from our paper.


## Overview
Multi-MoleScale addresses the challenge of integrating molecular graph structures with sequence information by combining:
- **Graph Contrastive Learning (GCL)**: Captures intrinsic graph-based features of molecules via multi-scale data augmentation.
- **BERT-based Sequence Learning**: Models contextual relationships in molecular SMILES sequences.
- **Co-attention Mechanism**: Fuses graph and sequence embeddings to enhance predictive performance.

The framework outperforms state-of-the-art methods on diverse datasets, including 12 molecular property datasets, ADMET datasets, and breast cancer cell line datasets.



## Dependencies
Ensure the following dependencies are installed:
- Python 3.8+
- PyTorch ≥1.2
- NumPy
- Scikit-learn
- TQDM
- RDKit (for molecular processing)
- DeepChem (for dataset handling)
- PyTorch Geometric (for graph neural networks)
- Pandas

Install dependencies via pip:
```bash
pip install torch numpy sklearn tqdm rdkit-pypi deepchem torch_geometric pandas
```


## File Structure
```
Multi-MoleScale/
├── data/                   # Datasets for training and evaluation
│   ├── pretrain-data.csv   # Data for graph pre-training (from ChEMBL)
│   ├── molecular_properties/  # 12 molecular property datasets (e.g., ESOL, BBBP)
│   ├── ADMET/              # ADMET prediction datasets
│   └── breast_cancer/      # Breast cancer cell line datasets
├── GCL-pretrain/           # Code for graph pre-training with GCL
│   ├── augmentations    # Multi-scale graph augmentation (ND, EP, FM, RW)
│   ├── eval        # GCL pre-training script
│   └── models      # GNN encoder for graph embeddings
├── PropertyPred.ipynb      # Main notebook for molecular property prediction
├── gclModel.rar            # Pre-trained GCL model (extract before use)
└── README.md               # This file
```


## Quick Start

### 1. Graph Pre-training (Optional)
To reproduce the GCL pre-training:
```bash
cd GCL-pretrain
python gcl_train.py --data ../data/pretrain-data.csv --output ../gclModel.rar
```
- Uses multi-scale data augmentation (Node Dropping, Edge Perturbation, Feature Masking, Random Walk Subgraphs).
- Saves the pre-trained model to `gclModel.rar`.


### 2. Molecular Property Prediction
Open `PropertyPred.ipynb` in Jupyter Notebook and follow these steps:
1. **Configure Paths**: Update file paths to point to your local dataset and model directories.
2. **Load Pre-trained Models**: Load the pre-trained GCL model (`gclModel.rar`) and BERT model.
3. **Run Experiments**: Execute the notebook to train and evaluate on:
   - Molecular property datasets (classification/regression tasks).
   - ADMET datasets.
   - Breast cancer cell line datasets.

The notebook includes code for:
- Data splitting (8:1:1 train/validation/test).
- Co-attention fusion of graph and sequence embeddings.
- Performance evaluation (ROC-AUC for classification, RMSE for regression).


## Datasets
- **Pretraining Data**: 1.04M compounds from the [ChEMBL database](https://www.ebi.ac.uk/chembl/).
- **Molecular Property Datasets**: 12 benchmark datasets (e.g., ESOL, FreeSolv, BBBP, Tox21) from [MoleculeNet](https://moleculenet.org/).
- **ADMET Datasets**: 15 datasets from [ADMETlab 2.0](https://admetmesh.scbdd.com/).
- **Breast Cancer Datasets**: 14 cell line datasets for phenotypic screening.


## Citation
If you use Multi-MoleScale in your research, please cite our paper:
```
Lou X, Cai J, Siu SWI. Multi-MoleScale: A Multi-Scale Approach for Molecular Property Prediction with Graph Contrastive and Sequence Learning. [Journal/Conference Name, Year].
```


## Contact
For questions or issues, please contact:
- Xinpo Lou: [louxinpo@qq.com](mailto:louxinpo@pp.com)



## License
This project is licensed under the MIT License. See `LICENSE` for details.