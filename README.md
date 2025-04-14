# Multi-MoleScale: A Multi-Scale Approach for Molecular Property Prediction with Graph Contrastive and Sequence Learning

##  Dependencies

Dependencies:

- python 3.8+
- pytorch >=1.2
- numpy
- sklearn
- tqdm
- rdkit
- deepchem
- torch_geometric
- pandas

##  How to run

"GCL-pretrain" is used to conduct graph pre-training experiments, and the dataset for graph pre-training is from data/pretrain-data.csv.
'PropertyPred.ipynb' is used to conduct experiment of molecular properties.

Remember to change all file path to the path in your system.

The graph pretrained model is saved in gclModel.rar. Refer to 'PropertyPred.ipynb' to load the model.

The three primary datasets utilized for molecular property prediction can be found in the "data" folder.

