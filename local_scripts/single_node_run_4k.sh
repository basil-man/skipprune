#!/bin/bash

# Getting the node names (simplified for single node)
head_node=$(hostname)  # Assume single node
head_node_ip=$(hostname -I | awk '{print $1}')  # Get the IP address

port=6499
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# Model and data configurations
LENGTH=4000
RUN_NAME=DeepSeek-R1-Distill-Qwen-1.5B-${LENGTH}
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# Using a single GPU for training
N_GPUS=1
TP=1  # No tensor parallelism needed for single GPU
MODEL_DIR=checkpoints/${RUN_NAME}
DATA_DIR=data/past_aime_amc/length${LENGTH}

# Set batch sizes for single GPU
BATCH_SIZE=4
ROLLOUT_BS=16
ROLLOUT_N=8

# No Ray cluster in single GPU setup
# Run the training script on the head node
echo "Starting training script on the HEAD node"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=768 \
    data.max_response_length=${LENGTH} \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$ROLLOUT_BS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    +actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$ROLLOUT_BS \
    reward_model.enable=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.default_local_dir=$MODEL_DIR \
    trainer.default_hdfs_dir=null \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_math' \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.multisample_val=True \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 \
    trainer.num_keep_checkpoint=20 \
    trainer.resume_checkpoint=True
