## Download the data and place them here:

[1D & 2D Burgers](https://drive.google.com/drive/folders/1YLsK5GkFpRvrUI4olSEBaz1Jo7T7lO0C) `burgers_neumann.mat` & `burgers2d.mat`
(provided by [SOL](https://github.com/liu-ziyuan-math/spectral_operator_learning))



[2D reaction-diffusion & 2D Darcy flow](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986) `2D_DarcyFlow_beta100.0_Train.hdf5` & `2D_diff-react_NA_NA.h5` (provided by [PDEBench](https://github.com/pdebench/PDEBench))


[corrected Pipe flow](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986](https://drive.google.com/drive/folders/1WPAs6bXttCPOWrDaUudC8B4dKPoju1OO)https://drive.google.com/drive/folders/1WPAs6bXttCPOWrDaUudC8B4dKPoju1OO) (the previous version is provided by [geo-FNO](https://github.com/neuraloperator/Geo-FNO))

## Corrected Pipe

The previous version of pipe dataset involves a few instances with artificial discontinuities at the upper edge of the outlet. These discontinuities arise from bugs in the numerical solver when handling the free BCs. So we recomputed the reference solutions to address this issue.
![image](https://github.com/liu-ziyuan-math/SPFNO/blob/main/data/corrected%20pipe/new-pipe.png)
