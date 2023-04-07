# SegFormer
SegFormer: A Topic Segmentation Model with Controllable Range of Attention

### Install: (We use the transformer framework and develop in an editable way.)  
cd .../transformers/  
pip install --editable .

### Hardware requirements:  
Quadro rtx8000*4 (Gpu memory: 48G).  
Reducing the blocks_size (16,24,32) and batch_size can reduce the use of GPU memory.  
Another way is to limit the maximum input length of articles in split_trainvaltest.py (we set the maximum input length=150).  
Changing the logging_steps and save_steps can accelerate the training speed.

### Software requirements:  
see requirements.txt

### Datasets:  
Download from https://github.com/sebastianarnold/WikiSection (use the .ref files)

### Pre-processing:
1.Set the dataset name in line#7 of split_trainvaltest.py  
2.python split_trainvaltest.py  
3.Put the generated folders under sequence-labeling/.  

### Train (The code will save the best pk checkpoint on the val dataset):
cd .../transformers/examples/pytorch/sequence-labeling/   
En_Disease:  
python -m torch.distributed.launch --nproc_per_node=4 run_language_modeling.py --output_dir=dir/   --model_type=bert   --model_name_or_path=bert-base-uncased   --do_train   --do_eval   --evaluate_during_training    --train_data_file=train_file_name/   --eval_data_file=val_file_name/  --line_by_line --block_size 48   --num_train_epochs 20   --learning_rate 1e-5   --warmup_steps 100   --logging_steps 5   --save_steps 5   --per_device_train_batch_size 2   --gradient_accumulation_steps 4   --overwrite_output_dir --evaluation_strategy=steps --per_device_eval_batch_size 64 --con_loss --yuzhi 0.5 --choice 0 --label_num 27 --save_total_limit 1

En_City:  
python -m torch.distributed.launch --nproc_per_node=4 run_language_modeling.py --output_dir=dir/   --model_type=bert   --model_name_or_path=bert-base-uncased   --do_train   --do_eval   --evaluate_during_training    --train_data_file=train_file_name/   --eval_data_file=val_file_name/  --line_by_line --block_size 48   --num_train_epochs 20   --learning_rate 1e-5   --warmup_steps 1000   --logging_steps 50   --save_steps 50   --per_device_train_batch_size 2   --gradient_accumulation_steps 4   --overwrite_output_dir --evaluation_strategy=steps --per_device_eval_batch_size 64 --con_loss --yuzhi 0.5 --choice 0 --label_num 30 --save_total_limit 1

De_Disease:  
python -m torch.distributed.launch --nproc_per_node=4 run_language_modeling.py --output_dir=dir/   --model_type=bert   --model_name_or_path=bert-base-german-cased   --do_train   --do_eval   --evaluate_during_training    --train_data_file=train_file_name/   --eval_data_file=val_file_name/  --line_by_line --block_size 48   --num_train_epochs 20   --learning_rate 1e-5   --warmup_steps 100   --logging_steps 5   --save_steps 5   --per_device_train_batch_size 2   --gradient_accumulation_steps 4   --overwrite_output_dir --evaluation_strategy=step --per_device_eval_batch_size 64 --con_loss --yuzhi 0.5 --choice 0 --label_num 25 --save_total_limit 1

De_City:  
python -m torch.distributed.launch --nproc_per_node=4 run_language_modeling.py --output_dir=dir/   --model_type=bert   --model_name_or_path=bert-base-german-cased   --do_train   --do_eval   --evaluate_during_training    --train_data_file=train_file_name/   --eval_data_file=val_file_name/  --line_by_line --block_size 48   --num_train_epochs 20   --learning_rate 1e-5   --warmup_steps 1000   --logging_steps 5   --save_steps 5   --per_device_train_batch_size 2   --gradient_accumulation_steps 4   --overwrite_output_dir --evaluation_strategy=step --per_device_eval_batch_size 64 --con_loss --yuzhi 0.5 --choice 0 --label_num 27 --save_total_limit 1

### Test:  
For En_Disease and En_City:  
python test.py --model_type=bert  --output_dir=dir --model_name_or_path=model_save_path  --do_eval --eval_data_file=test_file_path --line_by_line --block_size 64 --per_device_eval_batch_size 64 --English  

For De_Disease and De_City:  
python test.py --model_type=bert  --output_dir=dir --model_name_or_path=model_save_path  --do_eval --eval_data_file=test_file_path --line_by_line --block_size 64 --per_device_eval_batch_size 64  

### Contact by email:
haitao.bai@stu.xjtu.edu.cn

