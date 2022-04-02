# MTER

A parallel implementation of MTER based on the idea of parameter server. Number of processes can be configured for training efficiency based on computation capacity. 

## Usage

[Download](https://drive.google.com/drive/folders/1BYyyJW8BBl13KP4W9pxVCvw13CHVOrsV?usp=sharing "yelp data") the processed Yelp review dataset in a folder called `yelp_recursive_entry/`. 

The provided training and testing set are split from yelp_recursive.entry for testing the model. Split yelp_recursive.entry for different train/val/test settings. 

Train model: `python MTER_tripletensor_tucker.py --options XXX` 

| option        | default     | description |
| -----------   | ----------- | ----------- |
| -u, --useremb | 15       | user embedding dimension |
| -i, --itememb | 15       | item embedding dimension |
| -f, --featemb | 12       | feature embedding dimension |
| -w, --wordemb | 12       | opinion embedding dimension |
| --bpr | 10       | weight of BPR loss |
| --iter | 200,000       | number of training iterations |
| --lr | 0.1       | initial learning rate for adagrad |
| --nprocess | 4       | number of processes for multiprocessing |

The learned model parameters are stored in the folder 'results'. 

Please refer to our paper ['Explainable Recommendation via Multi-Task Learning in Opinionated Text Data'](https://dl.acm.org/citation.cfm?id=3210010) for more details.

## Citation
If you find this useful for your reserach, please consider cite:
```
@inproceedings{Wang:2018:ERV:3209978.3210010,
 author = {Wang, Nan and Wang, Hongning and Jia, Yiling and Yin, Yue},
 title = {Explainable Recommendation via Multi-Task Learning in Opinionated Text Data},
 booktitle = {The 41st International ACM SIGIR Conference on Research \&\#38; Development in Information Retrieval},
 series = {SIGIR '18},
 year = {2018},
 isbn = {978-1-4503-5657-2},
 location = {Ann Arbor, MI, USA},
 pages = {165--174},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3209978.3210010},
 doi = {10.1145/3209978.3210010},
 acmid = {3210010},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {explainable recommendation, multi-task learning, sentiment analysis, tensor decomposition},
} 
```
