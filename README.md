# LANTK
This is the code repository for the NeurIPS paper [Label-Aware Neural Tangent Kernel: Toward Better Generalization and Local Elasticity](https://arxiv.org/pdf/2010.11775.pdf).
If you use this code for your work, please cite
```

@article{chen2020label,
  title={Label-Aware Neural Tangent Kernel: Toward Better Generalization and Local Elasticity},
  author={Chen, Shuxiao and He, Hangfeng and Su, Weijie J},
  journal={NeurIPS},
  year={2020}
}

```



## Installing Dependencies
Use virtual environment tools (e.g miniconda) to install packages and run experiments\
python>=3.6\
[pytorch](https://pytorch.org)\
neural-tangents

## Code Organization

The code is organized as follows:
- CNN.py (CNN for binary classification)
- ntk_google.py (CNTK for binary classification)
- ntl_google.py (LANTK-HR for binary classification)
- CNN_multi.py (CNN for multi-class classification)
- ntk_google_multi.py (CNTK for multi-class classification)
- ntl_google_multi.py (LANTK-HR for multi-class classification)
- nn.py (2-layer NNs)
- ntk.py (NTK for 2-layer NNs)
- ntl.py (LANTK-HR for 2-layer NNs)
- ntk_approx.py (Approximate NTK $K_t^{(2)}$)
- ntk_simple_accelerate.py (LANTK-NTH)
- MSCOCO/label_dynamic.py (Dynamics of local elasticity over training)

## Reproducing experiments
To reproduce the experiments for CNN/CNTK/LANTK-HR on binary classification:
```
sh run_cnn_experiments.sh
sh run_ntk_google_experiments.sh
sh run_ntl_google_experiments.sh
```
Note that these commands are similar for CNN/CNTK/LANTK-HR on multi-class classification and NN/NTK/LANTK-HR for 2-layer NNs.

To reproduce the experiments for LANTK-NTH:
```
python ntl_simple_accelerate.py neg=3 pos=5 (an example)
```

To reproduce the experiments for dynamics of local elasticity over training:
```
CUDA_VISIBLE_DEVICES=1 python MSCOCO/labels_dynamic.py dataset=MSCOCO method_option=kernel 
model_option=MLPNet loss_option=MSE pos_one=dog pos_two=bench neg_one=cat neg_two=chair label_system=one
```
