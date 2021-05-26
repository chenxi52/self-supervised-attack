# Self-supervised Contrastive Learning Based Universarial Black-box Attack
intro:

### CPC from

Implementation of [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) Paper (https://arxiv.org/abs/1807.03748).

link[CPC github](https://github.com/vwrj/CPC/tree/f263b969a1b41ba761342633fe46f0b8148a3212)

### Preparing

- Download ILSVRC2012_img_val and save it in `.\data` dir

- The choosen validation set is written in `selected_data.csv`
```
|-data
  |-ILSVRC2012_img_val
  |  |-n01484850
  |  |-...
  |  └─...
  └─selected_data.csv   
```

### Usage

- Run `train.py` file to train the CPC encoder and save weight.
- Run `CpcAttack.py` to generate adv examples and evaluate them.
  
  e.g. run the attack 14 iterations using momentum 1.1 on gpu3 and evaluate on model vgg19_bn
```
  python CpcAttack.py --model vgg19_bn --iters 14 --gpu 3 --momentum 1.1
```


