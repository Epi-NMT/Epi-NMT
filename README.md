# Epi-NMT
Code for paper:

Epi-NMT: Episodic Training for Low-Resource Domain Adaptation in Neural Machine Translation

> To download the data

The data are from https://opus.nlpl.eu/. 

For your convenience, you can download them via the google drive: 

https://drive.google.com/file/d/1CRA5YK9vBT86a7jbQTc3_jfBZ0PDJJtb/view?usp=sharing

Place the file at the same directory of our .py files.

> To reproduce the paper

  - Implementation environment: 
    - Python version: 3.7
    - Pytorch: 1.10.2
    - transformers: 4.17.0
    - datasets: 2.0.0
    - sacrebleu: 2.0.0

  - Transfer data from txt to json, run:
  
      python create_data.py 
      
  - Train Epi-NMT:
  
      python Epi_NMT_train.py --gpu 0 --batchsz 54 --ds_batchsz 54 --alpha 0.3 --warmup_iterations 3000 --test_every 3
      
  Note: Tune the batchsz and ds_batchsz according to your own device.
  
   A single RTX 3090 GPU spent 30+ hours for training. 
    
