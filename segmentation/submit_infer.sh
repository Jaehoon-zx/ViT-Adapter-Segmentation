#!/bin/bash

#SBATCH --job-name=ViT_Adapter # Submit a job named "example"
#SBATCH --output=infer_1.txt  # 스크립트 실행 결과 std output을 저장할 파일 이름

#SBATCH --gres=gpu:1          # Use 1 GPU
#SBATCH --time=0-12:00:00        # d-hh:mm:ss 형식, 본인 job의 max time limit 지정
#SBATCH --mem=20000              # cpu memory size
#SBATCH --cpus-per-task=16       # cpu 개수

# module purge
# module load cuda/11.3            # 필요한 쿠다 버전 로드
# eval "$(conda shell.bash hook)"  # Initialize Conda Environment
# conda activate fourier       # Activate your conda environment

source /home/${USER}/.bashrc

srun python image_demo.py \
        configs/pascal_context/mask2former_beit_adapter_base_480_40k_pascal_context_59_ss.py \
        test_1/best_mIoU_iter_4000.pth \
        MODIS_dataset_sample/images/ \
        --palette ade20k \