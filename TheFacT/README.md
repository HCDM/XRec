# The FacT

This is an implementation for the paper titled "The FacT: Taming Latent Factor Models for Explainability with Factorization Trees" which is published at SIGIR 2019. We public our source code in this repository.

The slides of our oral presenation in SIGIR 2019 can be found [here](http://www.cs.virginia.edu/~yj9xs/).

### Algorithm
The FacT model aims at explaining latent factor based recommendation algorithms with rule-based explanations. It integrates regression trees to guide the learning of latent factor models for recommendation, and uses the learned tree structure to explain the resulting latent factors. With user-generated reviews, regression trees on users and items are built respectively, and each node on the trees are asscoiated with a latent profile to represent users and items. The detailed algorithm can be found in the [paper](https://arxiv.org/pdf/1906.02037.pdf).

### Usage
To run the code to generate experimental results like those found in our papers, you will need to run a command in the following format, using Python 2:
```
$ cd code
$ python main.py [-h] [--train_file] [--test_file] [--num_dim NUM_DIM]  
                 [--max_depth MAX_DEPTH] [--lambda_u LAMBDA_U] [--lambda_v LAMBDA_V]
                 [--lambda_bpr LAMBDA_BPR] [--num_BPRpairs NUM_BPRPAIRS]
                 [--batch_size BATCH_SIZE] [--learning_rate lr] [--num_run NUM_RUN]
                 [--num_iter_user NUM_ITER_USER] [--num_iter_item NUM_ITER_ITEM]
                 [--random_seed] 
```
The results will be stored in ./results/

In our papers, we used two widely used benchmark datasets collected from [Amazon](http://jmcauley.ucsd.edu/data/amazon) and [yelp](https://www.yelp.com/dataset). The files in data folder are examples of preprocessed dataset with extracted feature opinion scrores. The data format is:
```
user_id, item_id, rating, [list of feature opinions]
```
Example:  
1, 0, 4, 1 1 2 1  
user_id = 1, item_id = 0, rating = 4, rating for feature 1 = 1, rating for feature 2 = 1.

### Citation
If you use this code to produce results for your scientific publication, please refer to our SIGIR 2019 paper:
```
@inproceedings{tao2019the,
	title="The FacT: Taming Latent Factor Models for Explainability with Factorization Trees",
	author="Yiyi {Tao} and Yiling {Jia} and Nan {Wang} and Aobo {Yang} and Hongning {Wang}",
	booktitle="SIGIR 2019 : 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval",
	year="2019"
}

```
