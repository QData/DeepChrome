# AttentiveChrome

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

AttentiveChrome is a unified architecture to model and to interpret dependencies among chromatin factors for controlling gene regulation. AttentiveChrome uses a hierarchy of multiple Long short-term memory (LSTM) modules to encode the input signals and to model how various chromatin marks cooperate automatically. AttentiveChrome trains two levels of attention jointly with the target prediction, enabling it to attend differentially to relevant marks and to locate important positions per mark. We evaluate the model across 56 different cell types (tasks) in human. Not only is the proposed architecture more accurate, but its attention scores also provide a better interpretation than state-of-the-art feature visualization methods such as saliency map. 

**Feature Generation for AttentiveChrome model:** 

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

**Datasets**

We have provided a toy dataset to test out model in the data subdirectory of v2PyTorch

The complete set of 56 Cell Type datasets is located at https://zenodo.org/record/2652278

The rows are bins for all genes (100 rows per gene) and the columns are organised as follows:

GeneID, Bin ID, H3K27me3 count, H3K36me3 count, H3K4me1 count, H3K4me3 count, H3K9me3 counts, Binary Label for gene expression (0/1)  
e.g. 000003,1,4,3,0,8,4,1

**Running The Model** 

See the v1LuaTorch or v2PyTorch directories to run the code.



# v2PyTorch folder includes Pytorch version of the  AttentiveChrome Implementation. 
You can run it via the following command: 

```
python train.py --cell_type Toy
```



## We also provide trained AttentiveChrome models through the Kipoi model zoo     [http://kipoi.org/](http://kipoi.org/)

Attentive Chrome model can be run using Kipoi, which is a repository of predictive models for genomics. All models in the repo can be used through shared API.

- The utility codes to adapt AttentiveChrome to Kipoi are in /kipoiutil

### Installation Requirements
* python>=3.5
* numpy
* pytorch-cpu
* torchvision-cpu

## Quick Start
### Creating new conda environtment using kipoi
`kipoi env create AttentiveChrome`


### Activating environment
`conda activate kipoi-AttentiveChrome`

## Command Line
We can run AttentiveChrome using a terminal.

### Getting example input file
To get an example input file for a specific model, run the following command. Replace {model_name} with the actual name of model (e.g. E003, E005, etc.)

`kipoi get-example AttentiveChrome/{model_name} -o example_file`

example: `kipoi get-example AttentiveChrome/E003 -o example_file`

### Predicting using example file
To make a prediction using an input file, run the following command.

`kipoi predict AttentiveChrome/{model_name} --dataloader_args='{"input_file": "example_file/input_file", "bin_size": 100}' -o example_predict.tsv`

This should produce a tsv file containing the results. To run it using another file, replace "example_file/input+file" with the path of your file.

## Python API
We can also use Attentive Chrome through the Kipoi Python API.
### Fetching the model
First, import kipoi:
`import kipoi`

Next, get the model. Replace {model_name} with the actual name of model (e.g. E003, E005, etc.)

`model = kipoi.get_model("AttentiveChrome/{model_name}")`

### Predicting using pipeline
`prediction = model.pipeline.predict({"input_file": "path to input file", "bin_size": {some integer}})`

This returns a numpy array containing the output from the final softmax function.

e.g. `model.pipeline.predict({"input_file": "data/input_file", "bin_size": 100})`

### Predicting for a single batch
First, we need to set up our dataloader `dl`.

`dl = model.default_dataloader(input_file="path to input file", bin_size={some integer})`

Next, we can use the iterator functionality of the dataloader.

`it = dl.batch_iter(batch_size=32)`

`single_batch = next(it)`

First line gets us an iterator named `it` with each batch containing 32 items. We can use `next(it)` to get a batch.

Then, we can perform prediction on this single batch.

`prediction = model.predict_on_batch(single_batch['inputs'])`

This also returns a numpy array containing the output from the final softmax function.


# We have extended attentiveChrome to DeepDiffChrome


- [DeepDiff: Deep-learning for predicting Differential
gene expression from histone modifications](https://academic.oup.com/bioinformatics/article/34/17/i891/5093224)

- Code Github [https://github.com/QData/DeepDiffChrome](https://github.com/QData/DeepDiffChrome)


## Meanwhile, here are some links for general data processing tools/guidance on ChIP-seq data:

[https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003326](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003326)

[https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5389943/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5389943/)

[https://bedtools.readthedocs.io/en/latest/](https://bedtools.readthedocs.io/en/latest/)
