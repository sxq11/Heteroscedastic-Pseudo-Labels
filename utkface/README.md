# UTKFace Dataset

## Dataset
#### Researchers can get the UTKFace dataset from https://susanqq.github.io/UTKFace/ (Aligned&Cropped Faces). Extract the zip file and set up the files according to the example files in data/UTKFace

# Dependencies

```bash
conda env create -f environment.yml
conda activate hpl
```

# Train

```bash
python main_ours.py --data_dir <path_to_data_dir> --output_dir <path_to_output_dir> --lr 1e-4 --fc_lr 1e-3 --meta_lr 1e-4 --num_epochs 30 --batch_size 32
```