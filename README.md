# DeepChrome

Go to Vizualization Website: [DeepChrome visualization website](www.qdata.cs.virginia.edu)

Reference Paper: [DeepChrome: Deep-learning for predicting gene expression from histone modifications](https://arxiv.org/abs/1607.02078)

DeepChrome is a unified CNN framework that automatically learns combinatorial interactions among histone modification marks to predict the gene expression. It is able to handle all the bins together, capturing both neighboring range and long range interactions among input features, as well as automatically extract important features. In order to interpret what is learned, and understand the interactions among histone marks for prediction, we also implement an optimizationbased technique for visualizing combinatorial relationships from the
learnt deep models. Through the CNN model, DeepChrome incorporates representations of both local neighboring bins as well as the whole gene.

**Feature Generation for DeepChrome model:** 
Bins of length 100 base-pairs (bp) are selected from regions (+/- 5000 bp) flanking the transcription start site (TSS) of each gene. The signal value of all five selected histone modifications from REMC in bins forms input matrix X, while discretized gene expression (label +1/-1) is the output y.

Toy dataset has been provided inside data folder.

To perform training : 
```
th doall.lua
```
To perform testing/Get visualization output: 
```
the doall_eval.lua
```
