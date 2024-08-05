# ECG Analysis with TCN and BiLSTM

This project uses Temporal Convolutional Networks (TCN) and Bidirectional Long Short-Term Memory Networks (BiLSTM) to analyze ECG data.

## Requirements
* Python 3.11
* tensorflow/tensorflow-gpu
* numpy
* scipy
* scikit-learn
* matplotlib
* imbalanced-learn (0.4.3)
* SMOTE
* torch
* seaborn
* argparse

## dataset
We evaluated our model using [the PhysioNet MIT-BIH Arrhythmia database](https://www.physionet.org/physiobank/database/mitdb/)
The dataset is then preprocessed by label matching.
Download links to the data used in this project（https://zhengyu.tech/nextcloud/s/96woibegWtiwkrr）

## train
```
python TCN+BiLSTM.py --data_dir data/mitbih_1 --epochs 30
```
## Citation
If you find it useful, please cite our paper as follows:

```
@article{mousavi2018inter,
  title={Inter-and intra-patient ECG heartbeat classification for arrhythmia detection: a sequence to sequence deep learning approach},
  author={Mousavi, Sajad and Afghah, Fatemeh},
  journal={arXiv preprint arXiv:1812.07421},
  year={2018}
}
```
## Licence 
For academtic and non-commercial usage 
