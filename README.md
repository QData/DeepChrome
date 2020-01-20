# DeepChrome


### Reference Paper: [DeepChrome: Deep-learning for predicting gene expression from histone modifications](http://bioinformatics.oxfordjournals.org/content/32/17/i639.abstract)

BibTex Citation:
```
@article{singh2016deepchrome,
  title={DeepChrome: deep-learning for predicting gene expression from histone modifications},
  author={Singh, Ritambhara and Lanchantin, Jack and Robins, Gabriel and Qi, Yanjun},
  journal={Bioinformatics},
  volume={32},
  number={17},
  pages={i639--i648},
  year={2016},
  publisher={Oxford Univ Press}
}
```

DeepChrome is a unified CNN framework that automatically learns combinatorial interactions among histone modification marks to predict the gene expression. It is able to handle all the bins together, capturing both neighboring range and long range interactions among input features, as well as automatically extract important features. In order to interpret what is learned, and understand the interactions among histone marks for prediction, we also implement an optimizationbased technique for visualizing combinatorial relationships from the
learnt deep models. Through the CNN model, DeepChrome incorporates representations of both local neighboring bins as well as the whole gene.


**Feature Generation for DeepChrome model:** 

We used the five core histone modification (listed in the paper) read counts from REMC database as input matrix. We downloaded the files from [REMC dabase](http://egg2.wustl.edu/roadmap/web_portal/processed_data.html#ChipSeq_DNaseSeq). We converted 'tagalign.gz' format to 'bam' by using the command:
```
gunzip <filename>.tagAlign.gz
bedtools bedtobam -i <filename>.tagAlign -g hg19chrom.sizes > <filename>.bam 
```
Next, we used "bedtools multicov" to get the read counts. 
Bins of length 100 base-pairs (bp) are selected from regions (+/- 5000 bp) flanking the transcription start site (TSS) of each gene. The signal value of all five selected histone modifications from REMC in bins forms input matrix X, while discretized gene expression (label +1/-1) is the output y.

For gene expression, we used the RPKM read count files available in REMC database. We took the median of the RPKM read counts as threshold for assigning binary labels (-1: gene low, +1: gene high). 

We divided the genes into 3 separate sets for training, validation and testing. It was a simple file split resulting into 6601, 6601 and 6600 genes respectively. 

We performed training and validation on the first 2 sets and then reported AUC scores of best performing epoch model for the third test data set. 

Toy dataset has been provided inside "code/data" folder.

After downloading "code/" folder:

To perform training : 
```
th doall.lua
```
To perform testing/Get visualization output: 
```
the doall_eval.lua
```

The complete set of 56 Cell Type datasets is located at https://zenodo.org/record/2652278

The rows are bins for all genes (100 rows per gene) and the columns are organised as follows:

GeneID, Bin ID, H3K27me3 count, H3K36me3 count, H3K4me1 count, H3K4me3 count, H3K9me3 counts, Binary Label for gene expression (0/1)  
e.g. 000003,1,4,3,0,8,4,1

# AttentiveChrome

### We have extended DeepChrome to AttentiveChrome @ 

[https://github.com/QData/AttentiveChrome](https://github.com/QData/AttentiveChrome)

+ AttentiveChrome is a unified architecture to model and to interpret dependencies among chromatin factors for controlling gene regulation. AttentiveChrome uses a hierarchy of multiple Long short-term memory (LSTM) modules to encode the input signals and to model how various chromatin marks cooperate automatically. AttentiveChrome trains two levels of attention jointly with the target prediction, enabling it to attend differentially to relevant marks and to locate important positions per mark. We evaluate the model across 56 different cell types (tasks) in human. Not only is the proposed architecture more accurate, but its attention scores also provide a better interpretation than state-of-the-art feature visualization methods such as saliency map.

+ A copy of AttentiveChrome Code is also added as subfolder ./AttentiveChrome-PyTorch in this repo. 

Reference Paper: [Attend and Predict: Using Deep Attention Model to Understand Gene Regulation by Selective Attention on Chromatin](https://arxiv.org/abs/1708.00339)

BibTex Citation:
```
@inproceedings{singh2017attend,
  title={Attend and Predict: Understanding Gene Regulation by Selective Attention on Chromatin},
  author={Singh, Ritambhara and Lanchantin, Jack and Sekhon, Arshdeep  and Qi, Yanjun},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6769--6779},
  year={2017}
}
```

## We also provide trained AttentiveChrome models through the Kipoi model zoo     [http://kipoi.org/](http://kipoi.org/)

Attentive Chrome model can be run using Kipoi, which is a repository of predictive models for genomics. All models in the repo can be used through shared API.

- The utility codes to adapt AttentiveChrome to Kipoi are in ./AttentiveChrome-PyTorch/kipoiutil
