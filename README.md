# Improving Pedestrian Attribute Recognition With Weakly-Supervised Multi-Scale Attribute-Specific Localization

Code for the paper "Improving Pedestrian Attribute Recognition With Weakly-Supervised Multi-Scale Attribute-Specific Localization", ICCV 2019, Seoul.

[[Paper]](https://arxiv.org/abs/1910.04562) [[Poster]](https://chufengt.github.io/publication/pedestrian-attribute/iccv_poster_id2029.pdf)

Contact: chufeng.t@foxmail.com or tcf18@mails.tsinghua.edu.cn

## Environment

- Python 3.6+
- PyTorch 0.4+

## Datasets

- RAP: http://rap.idealtest.org/
- PETA: http://mmlab.ie.cuhk.edu.hk/projects/PETA.html
- PA-100K: https://github.com/xh-liu/HydraPlus-Net

The original datasets should be processed to match the DataLoader.

We provide the label lists for training and testing.

## Training and Testing

```
python main.py --approach=inception_iccv --experiment=rap
```

```
python main.py --approach=inception_iccv --experiment=rap -e --resume='model_path'
```

## Pretrained Models

We provide the pretrained models for reference, the results may slightly different with the values reported in our paper.

| Dataset | mA    | Link                                                         |
| ------- | ----- | ------------------------------------------------------------ |
| PETA    | 86.34 | [Model](https://drive.google.com/file/d/1cvX43Qn_vydzT_jnmgwYUUe9hIA161PH/view?usp=sharing) |
| RAP     | 81.86 | [Model](https://drive.google.com/file/d/15paMK0-rKDsuzptDPK5kH2JuL8QO0HyS/view?usp=sharing) |
| PA-100K | 80.45 | [Model](https://drive.google.com/file/d/1xIw3jpvE1pDC3U464kcFJ58iSKCRNQ63/view?usp=sharing) |

## Reference

If this work is useful to your research, please cite:

```
@inproceedings{tang2019improving,
  title={Improving Pedestrian Attribute Recognition With Weakly-Supervised Multi-Scale Attribute-Specific Localization},
  author={Tang, Chufeng and Sheng, Lu and Zhang, Zhaoxiang and Hu, Xiaolin},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={4997--5006},
  year={2019}
}
```
