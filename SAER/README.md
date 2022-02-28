# Sentiment Aligned Explainable Recommendation

This repo contains the PyTorch implementation of the model Sentiment Aligned Explainable Recommendation (SAER) proposed in our WSDM 2021 paper ["Explanation as a Defense of Recommendation"](https://arxiv.org/abs/2101.09656). Please refer to the paper for the details of the algorithm.

## Dependency

All the depencies are included in the file `environment.yml`. If you use `conda`, installation can be done with the following command:
```
conda env create --name saer_env --file=environments.yml
```

## Usages

### Preprocess

Please refer to the data preprocessing [README](https://github.com/aobo-y/data/README.md) and follow the steps to prepare the data.

### Config

Check the configurations under the folder `/config`. Create a new or update existing model configuration.

### Train

Train the model specified by the given configuration file and optionally resume from an existing checkpoint

```
python train.py -m=saer --checkpoint=10
```

### Decode

Specify a text decoding strategy to generate recommendation explanations for the testing data with a trained model of a checkpoint, and save the output to a file for later evaluation

```
python decode.py -m=saer --checkpoint=20 --search=greedy --output=sear_decode.txt
```

### Evaluate

Evaluate the model on some metrics on-the-fly
```
python eval.py -m=saer --checkpoint=20 rmse mae ...
```

Directly evaluate the previously decoded output
```
python eval_decode.py -f=saer_decode.txt rmse bleu ...
```

## Reference
Please cite our paper if you use this code in your research:
```
@inproceedings{yang2021explanation,
  title={Explanation as a Defense of Recommendation},
  author={Yang, Aobo and Wang, Nan and Deng, Hongbo and Wang, Hongning},
  booktitle={In Proceedings of the Fourteenth ACM International Conference on Web Search and Data Mining (WSDM ’21)},
  year={2021}
}
```
