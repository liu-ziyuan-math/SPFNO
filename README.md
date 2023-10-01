# Technical details for SPFNO: SPECTRAL OPERATER LEARNING FOR PDES WITH DIRICHLET AND NEUMANN BOUNDARY CONDITIONS

Generally speaking, we follow the following criterion when selecting hyperparameters for benchmark:

1.If the strucutre is similar, choose the same `modes` and `width`. For example, FNO, OPNO, and SPFNO are all based on spectral method.

2.If the hyperparameters are given by the original paper, we will follow them if possible, e.g., LSM and Unet. Please refer to the source(`src`).

3.The amount of parameters are compatible so that the comparison is fair.

## Example 1: 1D Burgers
Under the instrcution of paper [OPNO](https://github.com/liu-ziyuan-math/spectral_operator_learning) : 

using Adam optimizer to train for `5000 epochs` with an initial `learning_rate of 0.001` that is halved every `500` epochs.

|       | width | modes | bandwidth | #Param | src |
|-------|-------|-------|-----------|--------|---|
| FNO   | 50    | 20X2  | 1         | 0.83m  | [OPNO](https://github.com/liu-ziyuan-math/spectral_operator_learning) |
| Unet  | -     | -     | -         | 2.7m   | [PDEBench](https://github.com/pdebench/PDEBench) |
| LSM   | 32    | *20*    | -         | 3.8m   |  |
| OPNO  | 50    | 40    | 3         | 1.5m   | [OPNO](https://github.com/liu-ziyuan-math/spectral_operator_learning) |
| SPFNO | 50    | 20    | 4         | 0.82m  |   |

## Example 2: 2D Burgers
Under the instrcution of paper of [OPNO](https://github.com/liu-ziyuan-math/spectral_operator_learning) : 

using Adam optimizer to train for `3000 epochs` with an initial `learning rate of 0.001` that is halved every `300` epochs.

|       | width | modes | bandwidth | #Param | src |
|-------|-------|-------|-----------|--------|---|
| FNO   | 24    | 8X2  | 1         | 0.60m  | [OPNO](https://github.com/liu-ziyuan-math/spectral_operator_learning) |
| Unet  | -     | -     | -         | 7.8m   | [PDEBench](https://github.com/pdebench/PDEBench) |
| LSM   | 32    | *12*    | -         | 4.8m   | [LSM](https://github.com/thuml/Latent-Spectral-Models) |
| OPNO  | 24    | 16    | 3         | 5.5m   | [OPNO](https://github.com/liu-ziyuan-math/spectral_operator_learning) |
| SPFNO | 24    | 16    | 4         | 9.7m  |   |

## Example 3: 2D Reaction-Diffusion(Allen Cahn)
Under the instrcution of [PDEBench](https://github.com/pdebench/PDEBench/blob/main/pdebench/models/config/args/config_diff-react.yaml)

using Adam optimizer to train for `500 epochs` with an initial `learning rate of 0.001` that is halved every `100` epochs.
