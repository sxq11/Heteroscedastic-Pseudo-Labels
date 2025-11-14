# IMDB-WIKI Dataset

## Dataset
####  Download and extract IMDB faces and WIKI faces respectively using

```bash
python download_imdb_wiki.py
```
#### We have provided required IMDB-WIKI-DIR file `imdb_wiki.csv` to set up balanced val/test set in folder `./data`. To reproduce the results in the paper, please directly use this file. You can also generate it using

```bash
python data/create_imdb_wiki.py
python data/preprocess_imdb_wiki.py
```

# Dependencies

```bash
conda env create -f environment.yml
conda activate hpl
```

# Train

```bash
python main_ours.py --data_dir <path_to_data_dir> --output_dir <path_to_output_dir> --lr 1e-4 --fc_lr 1e-3 --unc_lr 1e-4 --num_epochs 30 --batch_size 48
```