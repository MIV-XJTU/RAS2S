# 【ECCV 2024】Region-Aware Sequence-to-Sequence Learning for Hyperspectral Denoising

The official PyTorch implementation of our RAS2S(https://link.springer.com/chapter/10.1007/978-3-031-73027-6_13)

## Model
The primary implementation of the RAS2S can be found in the following directory:

```
 basic/models/competing_methods/s2s_backbone
```

Other competing methods can also be placed in the same folder.

## Dataset

### Download
#### The entire ICVL dataset download link: https://icvl.cs.bgu.ac.il/hyperspectral/

### Processing

#### The training and testing samples are listed in the following file:
```
 basic/utility/icvl_train_list.txt

 basic/utility/icvl_test_list.txt
```
#### Prepare the LMDB dataset for training.

```
cd basic
python utility/lmdb_data.py
```
#### Prepare the MAT dataset for testing.

```
cd basic
python utility/mat_data.py
```

## Training and Testing
### Configuration modification
Once the training and testing datasets are prepared, please ensure that the relevant data paths **trainDir, valDir**, and **testDir** are correctly set in the configuration file (**options/s2s_hsid.yml**).


### Training
Ensure that the model checkpoint path (**save_path**) is correctly set in the configuration file.

```
cd basic
python run.py -method s2s -mode train
```


### Testing
Please modify the model weight path (**resumePath**) in the configuration file. 

```
cd basic
python run.py -method s2s -mode test
```

## Citation

If any parts of our paper and code help your research, please consider citing us and giving a star to our repository.

```
@inproceedings{xiao2024region,
  title={Region-Aware Sequence-to-Sequence Learning for Hyperspectral Denoising},
  author={Xiao, Jiahua and Liu, Yang and Wei, Xing},
  booktitle={European Conference on Computer Vision},
  pages={218--235},
  year={2024},
  organization={Springer}
}
```