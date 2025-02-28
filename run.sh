#!/bin/bash -l

#$ -P ivc-ml            # Specify the SCC project name you want to use
#$ -l h_rt=11:00:00     # Specify the hard time limit for the job
#$ -N cvcl-saycamllava  # Give job a name
#$ -j y                 # Merge the error and output streams into a single file
#$ -l gpus=1            # One GPU 
#$ -l gpu_c=8.0         # 8.0 compute capability
#$ -pe omp 10           # 10 CPU cores

cd /projectnb/ivc-ml/ac25/Baby\ LLaVA/multimodal-baby/
source .cvcl_env/bin/activate

# full training
# python training/train.py --batch_size=32 --gpus=1 --num_workers=8 --checkpoint_callback=False --logger=True --pretrained_cnn --max_epochs=5 --multiple_frames --augment_frames

# fast dev run
# standard model
# python train.py --batch_size=16 --gpus=1 --num_workers=8 --checkpoint_callback=False --logger=False --pretrained_cnn --multiple_frames --augment_frames --fast_dev_run --text_encoder="embedding" 

# allow fine-tuning
# python train.py --batch_size=16 --gpus=1 --num_workers=8 --checkpoint_callback=False --logger=False --pretrained_cnn --finetune_cnn --multiple_frames --augment_frames --fast_dev_run --text_encoder="embedding" 


#Use ./run.sh | tee log.txt if running manually
#Using args from saycam contrastive.py and any info found in supplementary material (they mostly align)
python train.py \
    --lambda_mm=1. \
    --lambda_lm=0. \
    --embedding_type="flat" \
    --text_encoder="embedding" \
    --embedding_dim=512 \
    --dropout_i=.5 \
    --dropout_o=.0 \
    --cnn_dino \
    --pretrained_cnn \
    --finetune_cnn \
    --multiple_frames \
    --augment_frames \
    --normalize_features \
    --fix_temperature \
    --temperature=0.07 \
    --gpus=1 \
    --num_workers=8 \
    --batch_size=8 \
    --drop_last \
    --optimizer="AdamW" \
    --lr=1e-4 \
    --lr_scheduler \
    --weight_decay=0.1 \
    --val_batch_size=16 \
    --eval_include_sos_eos \
    --seed=0 \
    --optimize_unused \
    --max_epochs=400 \
    --check_val_every_n_epoch=1 \
    --checkpoint_callback \
    --logger=False \
    --exp_name="experiment_full_2"