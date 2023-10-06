# RetroRanker

This repository is the official implementation of RetroRanker, a ranking model built upon the graph neural network to mitigate the frequency bias in predictions of existing retrosynthesis models through re-ranking.

![1675683100128](image/README/1675683100128.png)

## Setup

```
conda create -n retroranker -c conda-forge -c rdkit -y python=3.9 rdkit=2022.03.1 
conda activate retroranker
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install fairseq==0.12.2
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html
pip install dgllife ogb==1.3.4 pympler lmdb
pip install rxnmapper 
```


## Steps to reproduce results

We provide the full scripts to generate training/testing data from raw predictions in this repository.

### Step 0: Downloading and preparing the data

We provide our data, trained checkpoint, and re-ranking scores prediction at [here](https://drive.google.com/drive/folders/1rpjyDV0b1N9H4aU0xqHiilgNe91WoKNc?usp=sharing). You may download the data and unzip the three folders below the root directory of the project (`RetroRanker/data, RetroRanker/model, RetroRanker/output`).

```
wget https://bdmstorage.blob.core.windows.net/shared/data_model_output.tar.gz
tar -xvzf data_model_output.tar.gz
```

According to the file size limiations, we only provide the intermediate processing files (molecule graphs) on the test data, while you can generate the other processing files by the instructions below.

If you are aiming to reproduce the paper results on the USPTO-full, you may skip the training process and follow the testing section after correctly setting up the data.

## Training

### Step 1: Preprocessing the predicted output

```
# Turning the raw prediction files into structured and grouped inputs
python preprocess.py
```

The corresponding files are saved at  `data/$dataset/1_preprocess/`.

### Step 2: Adding atom mapping

```
# Adding atom mapping information via rxnmapper
bash script/mapping.sh AT
bash script/mapping.sh R-SMILES
```

The corresponding files are saved at `data/$dataset/2_mapping/`.

### Step 3: Generating molecule graphs

```
# Generating molecule graphs for future usage in GNN
# Please mind the difference on settings between AttentiveFP and Graphromer

dataset=$1 # AT or R-SMILES

chunk_id=$2  # range(total_chunks)
total_chunks=$3 # 5 for AttentiveFP and 30 for Graphormer
file_identifier=$4 # 0-7 or test

cd $ROOT/RetroRanker
python generate_graphs.py --dataset $dataset
--chunk_id${chunk_id} --total_chunks ${total_chunks}
--file_identifier ${file_identifier}
--save_type dgl (for AttentiveFP) or pyg (for Graphormer)
```

The corresponding files are saved at `data/$dataset/3_gengraph/`.

### Step 4: Training RetroRanker

#### AttentiveFP Backbone

```
# Training the AttentiveFP-based model
python train_model.py --dataset AT or R-SMILES
```

The checkpoints are saved at `model/$dataset/$dataset_AF.pt`

#### Graphromer Backbone

```
# Training the Graphormer-based model (cost more time than AttentiveFP)
sh scripts/train_graphormer.sh
```

The checkpoints are saved at `model/$dataset/gh/`

## Testing: Re-ranking with RetroRanker

### AttentiveFP Backbone

```
python test_model.py --dataset $dataset --testset $dataset
```

### Graphromer Backbone

```
sh scripts/eval.sh 
# you may change the $traindata & $testdata in the script
```

### The best improvement on USPTO-full

The best re-ranking results on USPTO-full are displayed on analysis_af.ipynb (for AttentiveFP backbone) & analysis_gh.ipynb (for Graphromer backbone).
