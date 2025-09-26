@echo off
setlocal enabledelayedexpansion

@REM 特征对比
python train_mtl.py --model meg --num_epoch 150 --features gfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
python train_mtl.py --model meg --num_epoch 150 --features mel --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
python train_mtl.py --model meg --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
python train_mtl.py --model meg --num_epoch 150 --features cqt --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
python train_mtl.py --model meg --num_epoch 150 --features mfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"

@REM 网络对比
python train_mtl.py --model mcl --num_epoch 150 --features stft --task_type mtl
python train_mtl.py --model mcl --num_epoch 150 --features gfcc --task_type mtl
python train_mtl.py --model meg --num_epoch 150 --features stft --task_type mtl
python train_mtl.py --model meg --num_epoch 150 --features gfcc --task_type mtl
python train_mtl.py --model densenet121 --num_epoch 150 --features stft --task_type classification --batch_size 16
python train_mtl.py --model resnet18 --num_epoch 150 --features stft --task_type classification --batch_size 16
python train_mtl.py --model mobilenetv2 --num_epoch 150 --features stft --task_type classification --batch_size 16
python train_mtl.py --model resnet50 --num_epoch 150 --features stft --task_type classification --batch_size 16
python train_mtl.py --model swin --num_epoch 150 --features stft --task_type classification --batch_size 16

@REM 交叉验证
python train_mtl.py --model meg --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
python train_mtl.py --model meg --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_2.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_2.txt"
python train_mtl.py --model meg --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_3.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_3.txt"
python train_mtl.py --model meg --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_4.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_4.txt"
python train_mtl.py --model meg --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_5.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_5.txt"
