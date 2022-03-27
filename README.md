# DeepCKI
## Description
This is a graph-based representation learning method for predicting cell-cytokine interaction. We use both network information and node attributes to improve the performance. Protein-protein interaction (PPIs) networks of cytokine are used to construct graphs, which are used to propagate node attribtues, according to the definition of graph convolutional networks.

We use amino acid sequence (CT encoding), subcellular location (bag-of-words encoding) and protein domains (bag-of-words encoding) as the node attributes (initial feature representation).


## Usage
### Requirements
- Python 3.8
- TensorFlow
- Keras
- networkx
- scipy
- numpy
- pickle
- scikit-learn
- pandas

### Data
You can download the data from here <a href="http://www.immunexpresso.org" target="_blank">data</a>. 

### Steps

#### Step2: run the model
> cd src/Graph2GO     
> python main.py    
> **Note there are several parameters can be tuned: --dataset_type, use different cell-cytokine dataset . --model,VGAE or VAE
