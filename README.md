# SiNC-rPPG:
# Non-Contrastive Unsupervised Learning of Physiological Signals from Video

## Paper accepted to the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2023

<figure>
  <img src="./teaser.png" style="width:100%">
  <figcaption>Figure 1: Overview of the SiNC framework for rPPG compared with traditional supervised and unsupervised learning. Supervised and contrastive losses use distance metrics to the ground truth or other samples. Our framework applies the loss directly to the prediction by shaping the frequency spectrum, and encouraging variance over a batch of inputs. Power outside of the bandlimits is penalized to learn invariances to irrelevant frequencies. Power within the bandlimits is encouraged to be sparsely distributed near the peak frequency.</figcaption>
</figure>

## Contents
* Training code is in src/train.py
* Testing code is in src/test.py
* Experiment config file is in src/args.py
* Loss functions are in src/utils/losses.py
* Model architectures are in src/models/
* Dataloaders are in src/datasets/
* TODO: code for dataset preprocessing

### Citation
If you use any part of our code or data, please cite our paper.
```
@inproceedings{speth2023sinc,
  title={Non-Contrastive Unsupervised Learning of Physiological Signals from Video},
  author={Speth, Jeremy and Vance, Nathan and Flynn, Patrick and Czajka, Adam},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023},
}
```
