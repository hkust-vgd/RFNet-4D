# RFNet-4D: Joint Object Reconstruction and Flow Estimation from 4D Point Clouds

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rfnet-4d-joint-object-reconstruction-and-flow/3d-human-reconstruction-on-dynamic-faust)](https://paperswithcode.com/sota/3d-human-reconstruction-on-dynamic-faust?p=rfnet-4d-joint-object-reconstruction-and-flow)

This is an implementation of RFNet-4D, a new network architecture that jointly reconstructs objects and their motion flows from 4D point clouds: 

**RFNet-4D: Joint Object Reconstruction and Flow Estimation from 4D Point Clouds** <br />
[Tuan-Anh Vu](https://tuananh1007.github.io), [Duc-Thanh Nguyen](https://ducthanhnguyen.weebly.com/), [Binh-Son Hua](https://sonhua.github.io/), [Quang-Hieu Pham](https://pqhieu.github.io/), [Sai-Kit Yeung](http://saikit.org/)

For more details, please check [[Paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136830036.pdf), [[Arxiv version]](https://arxiv.org/abs/2203.16482), and [[Project webpage]](https://tuananh1007.github.io/RFNet-4D/).

If you have any question, please contact Tuan-Anh Vu <tavu@connect.ust.hk>.


**The proposed architecture is shown as below:**

![a](images/overview.png)

## Installation
First you have to make sure that you have all dependencies in place.  You can create and activate an anaconda environment called `unflow` using

```
conda env create -f environment.yml
conda activate unflow
```
Next, compile the extension modules. You can do this via
```
python setup.py build_ext --inplace
```

## Demo

You can test our code on the provided input point cloud sequences in the `demo/` folder. To this end, simple run
```
python generate.py configs/demo.yaml
```
This script should create a folder `out/demo/` where the output is stored.


## Dataset

You can download the pre-processed data (~42 GB) using

```
bash scripts/download_data.sh
```

The script will download the point-based point-based data for the [Dynamic FAUST (D-FAUST)](http://dfaust.is.tue.mpg.de/) dataset to the `dataset/` folder. 

Download 2 registration files the `dataset/` folder: [Link1](https://download.is.tuebingen.mpg.de/download.php?domain=dfaust&sfile=registrations_m.hdf5), [Link2](https://download.is.tuebingen.mpg.de/download.php?domain=dfaust&sfile=registrations_f.hdf5)

Then go to the `dataset/` folder and run the script for sample code parsing these files:
```
python scripts/write_sequence_to_obj.py 
```

The processed data should have a folder with the following structure:   
___
your_dfaust_folder/  
| 50002_chicken_wings/  
&nbsp;&nbsp;&nbsp;&nbsp;| 00000.obj  
&nbsp;&nbsp;&nbsp;&nbsp;| 00001.obj  
&nbsp;&nbsp;&nbsp;&nbsp;| ...  
&nbsp;&nbsp;&nbsp;&nbsp;| 000215.obj  
| 50002_hips/  
&nbsp;&nbsp;&nbsp;&nbsp;| 00000.obj  
&nbsp;&nbsp;&nbsp;&nbsp;| ...  
| ...  
| 50027_shake_shoulders/  
&nbsp;&nbsp;&nbsp;&nbsp;| 00000.obj  
&nbsp;&nbsp;&nbsp;&nbsp;| ...  
___

## Training

To train a new network from scratch or continue the current training, run
```
python train.py configs/unflow.yaml
```
You can monitor the training process on http://localhost:6006 using tensorboard:
```
cd OUTPUT_DIR
tensorboard --logdir ./logs --port 6006
```
where you replace `OUTPUT_DIR` with the respective output directory. For available training options, please have a look at `config/default.yaml`. 

## Generation

To start the normal mesh generation process using a trained model, use

```
python generate.py configs/unflow.yaml
```

You can find the outputs in the `out/pointcloud` folder.

Please note that the config files *_pretrained.yaml are only for generation, not for training new models: when these configs are used for training, the model will be trained from scratch, but during inference our code will still use the pretrained model.

## Evaluation

You can evaluate the generated output of a model on the test set using

```
python eval.py configs/unflow.yaml
```
The evaluation results will be saved to pickle and csv files.


## Note

If you have enough RAM for preload all data to RAM (~150GB) for faster training, you should keep the default dataloading. Otherwise, please change **subseq_dataset** to **subseq_dataset_ram_mp** in the path `im2mesh/data/__init__.py`


## Acknowledgements

Most of the code is borrowed from [Occupancy Flow](https://github.com/autonomousvision/occupancy_flow), [LPDC-Net](https://github.com/Gorilla-Lab-SCUT/LPDC-Net).

## Citation

If you find our code or paper useful, please consider citing

    @inproceedings{tavu2022rfnet4d,
      title={RFNet-4D: Joint Object Reconstruction and Flow Estimation from 4D Point Clouds},
      author={Tuan-Anh Vu, Duc-Thanh Nguyen, Binh-Son Hua, Quang-Hieu Pham, Sai-Kit Yeung},
      booktitle={Proceedings of European Conference on Computer Vision (ECCV)},
      year={2022}
    }
