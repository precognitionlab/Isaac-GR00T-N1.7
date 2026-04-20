+ 微调GrootN1.7
    + fork 在precognitionlab下：`https://github.com/precognitionlab/Isaac-GR00T-N1.7`
```

    # 安装
        $ git submodule update --init --recursive
        junweil@office-precognition:~/projects/wbc_manipulation/Isaac-GR00T-N1.7$ uv sync --python 3.10

        # 还需要账号同意cosmos-reason条款:
            https://huggingface.co/nvidia/Cosmos-Reason2-2B
            Go to your Hugging Face account settings (Settings > Access Tokens) and generate a Read token if you don't already have one.

            uv pip install -U "huggingface_hub[cli]"

            uv run huggingface-cli login

            ~/Desktop/github_projects/huggingface_key.txt

        # 还要安装一些东西，

            $ uv pip install bitsandbytes

    # finetune!



        # 修改 gr00t/experiment/launch_finetune.py, 开启
            config.training.gradient_checkpointing = True
            config.training.deepspeed_stage = 3
            #config.training.optim = "adamw_torch"
            config.training.optim = "paged_adamw_8bit"

        2x4090 48GB, bs=128, 47.8GB/48GB

        junweil@office-precognition:~/projects/wbc_manipulation/Isaac-GR00T-N1.7$ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run torchrun --nproc_per_node=2 gr00t/experiment/launch_finetune.py      --base-model-path ../GR00T-N1.7-3B/      --dataset-path ~/.cache/huggingface/lerobot/junweiliang/wbc_pick_up_object_from_ground      --embodiment-tag NEW_EMBODIMENT      --modality-config-path my_configs/g1_dex3_gripper_homie.py      --save-total-limit 2      --learning_rate 1e-4      --save-steps 2000      --max-steps 20000      --use-wandb      --warmup_ratio 0.05      --weight_decay 1e-5      --global-batch-size 128    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08      --dataloader-num-workers 4      --output-dir experiments/my_wbc_pick_up_object_from_ground_bs128_s20k

```
