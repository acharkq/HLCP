# HLCP

Code for *Legal Cause Prediction with Inner Descriptions and Outer Hierarchies*

## Dependencies

* Python 3.6
* Tensorflow 1.08
* scikit-learn
* [THULAC.so](https://github.com/thunlp/THULAC.so) -- Download the model file and unzip it under the "bin" folder

## Datasets

Time flies and things change. I never expect this work to be published and thus lost the processed datasets.

I collect some civil data after the paper was accepted. You can download it at [百度网盘](https://pan.baidu.com/s/1iTwTAiRIz2yj6K7Hw8ZB-g) (access code: 3tty). To run a demo on the civil dataset, unzip the file in the "bin" folder and execute:

```python
python train_cail.py [gpu_num]
```

You can still get the FSC dataset at [here](https://github.com/thunlp/attribute_charge) and CAIL dataset at [here](https://github.com/thunlp/CAIL). But you will need to process them.

## Reference

If you use our code, please cite our paper:

```bib
@inproceedings{liu2019legal,
  title={Legal Cause Prediction with Inner Descriptions and Outer Hierarchies},
  author={Liu, Zhiyuan and Tu, Cunchao and Liu, Zhiyuan and Sun, Maosong},
  booktitle={CCL},
  year={2019}
}
```
