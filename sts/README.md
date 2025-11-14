# STS-B dataset

## Dataset
#### Download GloVe word embeddings (840B tokens, 300D vectors) using

```bash
python glove/download_glove.py
```

####  We have provided both original STS-B dataset and our created balanced STS-B-DIR dataset in folder `./glue_data/STS-B`. To reproduce the results in the paper, please use our created STS-B-DIR dataset. If you want to get different balanced splits(train_new/test_new/dev_new), you can run
```bash
python original/split.py
```

##  Dependencies

```bash
conda env create -f environment.yml
conda activate sts
```

## Train
```bash
python main_ours.py --data_dir --data_dir <path_to_data_dir> --output_dir <path_to_output_dir> --labeled_ratio 0.1 --lr 1e-4 --fc_lr 1e-3 --unc_lr 1e-4 --num_epochs 200 --batch_size 32
```


