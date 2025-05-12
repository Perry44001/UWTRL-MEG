@echo off
setlocal enabledelayedexpansion

@REM set FEATURES=mel gfcc stft cqt
@REM python train_mtl.py --model meg --num_epoch 150 --features gfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
@REM python train_mtl.py --model meg --num_epoch 150 --features gfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_2.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_2.txt"
@REM python train_mtl.py --model meg --num_epoch 150 --features gfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_3.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_3.txt"
@REM python train_mtl.py --model meg --num_epoch 150 --features gfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_4.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_4.txt"
@REM python train_mtl.py --model meg --num_epoch 150 --features gfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_5.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_5.txt"

@REM python train_mtl.py --model meg --num_epoch 150 --features mel --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
@REM python train_mtl.py --model meg --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
@REM python train_mtl.py --model meg --num_epoch 150 --features cqt --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
@REM python train_mtl.py --model meg --num_epoch 150 --features mfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"


python train_mtl.py --model meg --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
python train_mtl.py --model meg --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_2.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_2.txt"
python train_mtl.py --model meg --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_3.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_3.txt"
python train_mtl.py --model meg --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_4.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_4.txt"
python train_mtl.py --model meg --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_5.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_5.txt"

@REM python train_mtl.py --model megx --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
@REM python train_mtl.py --model megx --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_2.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_2.txt"
@REM python train_mtl.py --model megx --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_3.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_3.txt"
@REM python train_mtl.py --model megx --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_4.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_4.txt"
@REM python train_mtl.py --model megx --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_5.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_5.txt"


@REM python train_mtl.py --model meg_ori --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
@REM python train_mtl.py --model meg_ori --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_2.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_2.txt"
@REM python train_mtl.py --model meg_ori --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_3.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_3.txt"
@REM python train_mtl.py --model meg_ori --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_4.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_4.txt"
@REM python train_mtl.py --model meg_ori --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_5.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_5.txt"

@REM python train_mtl.py --model meg_e --num_epoch 150 --features gfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
@REM python train_mtl.py --model meg_e --num_epoch 150 --features gfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_2.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_2.txt"
@REM python train_mtl.py --model meg_e --num_epoch 150 --features gfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_3.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_3.txt"
@REM python train_mtl.py --model meg_e --num_epoch 150 --features gfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_4.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_4.txt"
@REM python train_mtl.py --model meg_e --num_epoch 150 --features gfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_5.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_5.txt"

@REM python train_mtl.py --model meg_e --num_epoch 150 --features mel --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
@REM python train_mtl.py --model meg_e --num_epoch 150 --features stft --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
@REM python train_mtl.py --model meg_e --num_epoch 150 --features cqt --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
@REM python train_mtl.py --model meg_e --num_epoch 150 --features mfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"

@REM python train_mtl.py --model meg_mix --num_epoch 150 --features stft gfcc mel cqt mfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_1.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_1.txt"
@REM python train_mtl.py --model meg_mix --num_epoch 150 --features stft gfcc mel cqt mfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_2.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_2.txt"
@REM python train_mtl.py --model meg_mix --num_epoch 150 --features stft gfcc mel cqt mfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_3.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_3.txt"
@REM python train_mtl.py --model meg_mix --num_epoch 150 --features stft gfcc mel cqt mfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_4.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_4.txt"
@REM python train_mtl.py --model meg_mix --num_epoch 150 --features stft gfcc mel cqt mfcc --task_type mtl --train_list_path  "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\train_list_5.txt" --test_list_path "E:\MTQP\wjy_codes\shipsear_5s_16k_ocnwav_Pos\test_list_5.txt"
