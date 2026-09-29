# ASE-net

**Research Code for CLIP-Based Cross-Modal Image–Text Retrieval**

## Project Structure

```text
ASE-net/
├── readme.md
└── ASE/
    ├── requirements.txt
    ├── dataset/
    │   ├── process_data.py      
    │   └── utils/
    ├── src/
    │   ├── train.py              
    │   ├── test.py               
    │   ├── text.py               
    │   ├── evaluation.py         
    │   ├── visualization.py      
    │   ├── data/                 
    │   ├── model/               
    │   ├── loss/                 
    │   ├── metrics/              
    │   ├── option/               
    │   └── utils/                
    └── tools/
        └── check_project.py      
```

## Data Preparation

### Directory Layout

```text
ASE/data/RSITMD/
├── imgs/
│   └── ...
└── processed_data/
    ├── train_save.json
    └── test_save.json
```

Specify the dataset root with `--dataroot`. Each `img_path` entry in the JSON annotations is relative to this directory.

## Training

The following example matches the current command-line interface. Prepare the data and pretrained models first:

```shell
python src/train.py --dataset RSITMD --dataroot ./data/RSITMD --pretrain_path ./pretrain --model_name ase_rsitmd --GPU_id 0 --img_text_logits 1 --Topk_Selection --image_fintune --text_fintune --batch_size 64 --epoch 60 --loss_type InfoNCE
```

Training outputs are stored in:

```text
ASE/checkpoints/RSITMD/ase_rsitmd/
├── log/train.log
└── breakpoint/
    ├── best_model
    ├── current_epoch
    ├── test_best
    └── id_loss            
```

## Evaluation

Use the same model name, backbone, and token selection settings as in training:

```shell
python src/test.py --mode test --dataset RSITMD --dataroot ./data/RSITMD --pretrain_path ./pretrain --model_name ase_rsitmd --GPU_id 0 --img_text_logits 1 --Topk_Selection --batch_size 64
```

The evaluation entry point loads `breakpoint/best_model` and `breakpoint/current_epoch`, then reports bidirectional retrieval metrics and inference timing.