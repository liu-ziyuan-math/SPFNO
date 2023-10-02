# Technical details for SPFNO: SPECTRAL OPERATER LEARNING FOR PDES WITH DIRICHLET AND NEUMANN BOUNDARY CONDITIONS

Generally speaking, we follow the following criterion when selecting hyperparameters for benchmark:

1. If the hyperparameters have been given by the original paper, we will follow them if possible, e.g., LSM and Unet. Please refer to the source column(`src`).
2. If the strucutre is similar, choose the same `modes` and `width`. For example, FNO, OPNO, and SPFNO are all based on spectral method.
3. The amount of parameters and time consumption are compatible so that the comparison is fair.

## Example 1: 1D Burgers
Under the instrcution of paper [OPNO](https://github.com/liu-ziyuan-math/spectral_operator_learning) : 

using Adam optimizer to train for `5000 epochs` with an initial `learning_rate of 0.001` that is halved every `500` epochs. A long-termed training makes sure that all models are fully trained.

|       | width | modes | bandwidth | #Param | #src |
|-------|-------|-------|-----------|--------|---|
| FNO   | 50    | 20X2  | 1         | 0.83m  | [OPNO](https://github.com/liu-ziyuan-math/spectral_operator_learning) |
| Unet  | -     | -     | -         | 2.7m   | [PDEBench](https://github.com/pdebench/PDEBench) |
| LSM   | 32    | 20    | -         | 3.8m   |  |
| OPNO  | 50    | 40    | 3         | 1.5m   | [OPNO](https://github.com/liu-ziyuan-math/spectral_operator_learning) |
| SPFNO | 50    | 20    | 4         | 0.82m  |   |

## Example 2: 2D Burgers
Under the instrcution of paper of [OPNO](https://github.com/liu-ziyuan-math/spectral_operator_learning) : 

using Adam optimizer to train for `3000 epochs` with an initial `learning rate of 0.001` that is halved every `300` epochs.

|       | width | modes | bandwidth | #Param | #src |
|-------|-------|-------|-----------|--------|---|
| FNO   | 24    | 8x2  | 1         | 0.60m  | [OPNO](https://github.com/liu-ziyuan-math/spectral_operator_learning) |
| Unet  | -     | -     | -         | 7.8m   | [PDEBench](https://github.com/pdebench/PDEBench) |
| LSM   | 32    | 12    | -         | 4.8m   | [LSM](https://github.com/thuml/Latent-Spectral-Models) |
| OPNO  | 24    | 16    | 3         | 5.5m   | [OPNO](https://github.com/liu-ziyuan-math/spectral_operator_learning) |
| SPFNO | 24    | 16    | 4         | 9.7m  |   |

## Example 3: 2D Reaction-Diffusion(Allen Cahn)
Under the instrcution of [PDEBench](https://github.com/pdebench/PDEBench/blob/main/pdebench/models/config/args/config_diff-react.yaml)

using Adam optimizer to train for `500 epochs` with an initial `learning rate of 0.001` that is halved every `100` epochs.

|       | width | modes | bandwidth | #Param | #src |
|-------|-------|-------|-----------|--------|---|
| FNO   | 32    | 12x2  | 1         | 1.3m  | [OPNO](https://github.com/liu-ziyuan-math/spectral_operator_learning) |
| Unet  | -     | -     | -         | 7.8m   | [PDEBench](https://github.com/pdebench/PDEBench) |
| LSM   | 16    | 12    | -         | 1.2m   | [LSM](https://github.com/thuml/Latent-Spectral-Models) |
| SPFNO | 32    | 24    | 1         | 1.4m  |   |

- For LSM, case of width=16 outperforms that of width=32; when width=64, an "out of memory" error is reported on A100 80G.

## Example 4: 2D Darcy Flow
Under the instruction of [PDEBench](https://github.com/pdebench/PDEBench/blob/main/pdebench/models/config/config_darcy.yaml)

- For baseline, use Adam optimizer to train for `500 epochs` with an initial `learning rate of 0.001` that is halved every `100` epochs.
- For SPFNO, use Adam optimizer to train for `500 epochs` with an initial `learning rate of 0.004` that is halved `every 50 epochs that no improvement is seen` to accelerate the training. See [ReduceLROnPlateau](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)
- The `weight_decay` is set to `1e-6`, resulting in improvements for **all** models.

|       | width | modes | bandwidth | #Param | #src |
|-------|-------|-------|-----------|--------|---|
| FNO   | 32    | 12x2  | 1         | 2.4m  | [OPNO](https://github.com/liu-ziyuan-math/spectral_operator_learning) |
| Unet  | -     | -     | -         | 7.8m   | [PDEBench](https://github.com/pdebench/PDEBench) |
| LSM   | 64    | 12    | -         | 19.2m   | [LSM](https://github.com/thuml/Latent-Spectral-Models) |
| SPFNO | 32    | 24    | 1         | 2.4m  |   |
