set -x

export WANDB_API_KEY=e01fd7fd194dc6f847f489dc93e608cd17b947c2

project_name='grad_sim'
exp_name='qwen4_base_ppo'
export RAY_DEBUG=legacy

RAY_DATA_HOME="/apdcephfs_gy2/share_302625456"
CKPTS_DIR="/apdcephfs_gy2/share_302625456/user/lfsong/outputs/${project_name}/${exp_name}"
DATA_DIR="${RAY_DATA_HOME}/data"
MODEL_PATH="/apdcephfs_gy2/share_302625456/user/lfsong/models/Qwen/Qwen3-4B"
CRITIC_PATH="/apdcephfs_gy2/share_302625456/user/lfsong/models/Qwen/Qwen3-4B"

use_dynamic_bsz=False
param_offload=True
optimizer_offload=True
gradient_ckpt=True 

max_length=3000  
train_rollout_n=32

# sampling parameters
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout

fsdp_size=1
# CUDA_VISIBLE_DEVICES=0 nohup python -m verl.trainer.main_ppo \
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.trainer.main_ppo \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=gae \
    algorithm.gamma=1 \
    algorithm.lam=1 \
    data.train_files="${DATA_DIR}/dapo-math-17k_dedup_1_every_32.parquet" \
    data.val_files="${DATA_DIR}/aime24.parquet" \
    data.train_batch_size=${fsdp_size} \
    data.max_prompt_length=${max_length} \
    data.max_response_length=${max_length} \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_length + max_prompt_len)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_length + max_prompt_len)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_length + max_prompt_len)) \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${fsdp_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=${param_offload} \
    actor_rollout_ref.ref.fsdp_config.optimizer_offload=${optimizer_offload} \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.ref.fsdp_config.use_orig_params=true \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=${gradient_ckpt} \
    actor_rollout_ref.rollout.n=${train_rollout_n} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${fsdp_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    critic.optim.lr=0.0 \
    critic.model.use_remove_padding=True \
    critic.model.path="${CRITIC_PATH}" \
    critic.model.enable_gradient_checkpointing=${gradient_ckpt} \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=${param_offload} \
    critic.model.fsdp_config.optimizer_offload=${optimizer_offload} \
    critic.forward_micro_batch_size_per_gpu=1 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${fsdp_size} \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.val_before_train=False \
    trainer.resume_mode=disable $@
