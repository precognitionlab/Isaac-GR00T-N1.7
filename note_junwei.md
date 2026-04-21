+ 微调GrootN1.7
    + fork 在precognitionlab下：`https://github.com/precognitionlab/Isaac-GR00T-N1.7`
```

    # 安装 (gpu3, office)
        $ sudo apt install git-lfs && git lfs install
        $ sudo apt-get update && sudo apt-get install -y ffmpeg
        $ git submodule update --init --recursive

            # gpu3 安装需要把 aarch64 的一些路径删除
            # gpu3速度慢，要清华园UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"

        junweil@office-precognition:~/projects/wbc_manipulation/Isaac-GR00T-N1.7$ uv sync --python 3.10

        # 还需要账号同意cosmos-reason条款:
            https://huggingface.co/nvidia/Cosmos-Reason2-2B
            Go to your Hugging Face account settings (Settings > Access Tokens) and generate a Read token if you don't already have one.

            uv pip install -U "huggingface_hub[cli]"

            uv run huggingface-cli login

                # gpu3网络有问题，还要加这个 export HF_ENDPOINT=https://hf-mirror.com

            ~/Desktop/github_projects/huggingface_key.txt

        # 还要安装一些东西，

            $ uv pip install bitsandbytes

    # finetune!

        # 修改 gr00t/experiment/launch_finetune.py, 开启
            config.training.gradient_checkpointing = True
            config.training.deepspeed_stage = 3
            #config.training.optim = "adamw_torch"
            config.training.optim = "paged_adamw_8bit"

        # 2x4090 48GB, bs=128, 47.8GB/48GB, 20k steps, 要14小时

            # data num worker最好设置6， 比4要快

        junweil@office-precognition:~/projects/wbc_manipulation/Isaac-GR00T-N1.7$ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run torchrun --nproc_per_node=2 gr00t/experiment/launch_finetune.py      --base-model-path ../GR00T-N1.7-3B/      --dataset-path ~/.cache/huggingface/lerobot/junweiliang/wbc_pick_up_object_from_ground      --embodiment-tag NEW_EMBODIMENT      --modality-config-path my_configs/g1_dex3_gripper_homie.py      --save-total-limit 2      --learning_rate 1e-4      --save-steps 2000      --max-steps 20000      --use-wandb      --warmup_ratio 0.05      --weight_decay 1e-5      --global-batch-size 128    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08      --dataloader-num-workers 6      --output-dir experiments/my_wbc_pick_up_object_from_ground_bs128_s20k

            # 1xA6000 48GB, bs=128 也可以跑，41 GB/48GB, 11小时, 搞笑，不要用多卡了
            # 1xA6000 48GB, bs=256 OOM
            # 2xA6000 48GB, bs=256 OOM, 代码可能有问题

    # [04/21/2026] 实验，使用新的g1_dex3_gripper_homie_v2.py，states与action输出保持一致, action chunk length=50。

        # bs=128, lr=5e-5, 20k step

            # 之前用的action chunk=16, 现在是50， 需要跑一个代码先，重新算数据集的统计信息
                # See: https://github.com/precognitionlab/Isaac-GR00T-N1.7/blob/main/getting_started/data_config.md#required-fields

                $ rm ~/.cache/huggingface/lerobot/junweiliang/wbc_pick_up_object_from_ground/meta/relative_stats.json

                junweil@office-precognition:~/projects/wbc_manipulation/Isaac-GR00T-N1.7$ uv run python gr00t/data/stats.py --dataset-path ~/.cache/huggingface/lerobot/junweiliang/wbc_pick_up_object_from_ground --embodiment-tag NEW_EMBODIMENT --modality-config-path my_configs/g1_dex3_gripper_homie_v2.py
                    Loaded modality config: my_configs/g1_dex3_gripper_homie_v2.py
                    Generating stats for /home/junweil/.cache/huggingface/lerobot/junweiliang/wbc_pick_up_object_from_ground



        junweil@office-precognition:~/projects/wbc_manipulation/Isaac-GR00T-N1.7$ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run torchrun --nproc_per_node=1 gr00t/experiment/launch_finetune.py      --base-model-path ../GR00T-N1.7-3B/      --dataset-path ~/.cache/huggingface/lerobot/junweiliang/wbc_pick_up_object_from_ground      --embodiment-tag NEW_EMBODIMENT      --modality-config-path my_configs/g1_dex3_gripper_homie_v2.py      --save-total-limit 1      --learning_rate 5e-5      --save-steps 2000      --max-steps 20000      --use-wandb      --warmup_ratio 0.05      --weight_decay 1e-5      --global-batch-size 128    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08      --dataloader-num-workers 6      --output-dir experiments/my_wbc_pick_up_object_from_ground_bs128_s20k_v2

```
