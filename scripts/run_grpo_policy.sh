#!/usr/bin/env bash
# GRPO + LoRA on Policy only; online falsification reward.
# Requires: conda activate verl, parquet in data/verl/, reward backend ready.
#
# Single-GPU smoke (HF reward, slow):
#   REWARD_BACKEND=hf TOTAL_EPOCHS=1 TRAIN_BATCH_SIZE=2 ROLLOUT_N=2 ./scripts/run_grpo_policy.sh
#
# Two-GPU (preferred):
#   CUDA_VISIBLE_DEVICES=1 ./scripts/serve_reward_llm.sh   # terminal A
#   CUDA_VISIBLE_DEVICES=0 REWARD_BACKEND=http ./scripts/run_grpo_policy.sh

set -xeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERL_ROOT="${VERL_ROOT:-$ROOT/third_party/verl}"
MODEL_PATH="${MODEL_PATH:-$ROOT/models}"
TRAIN_FILE="${TRAIN_FILE:-$ROOT/data/verl/train.parquet}"
VAL_FILE="${VAL_FILE:-$ROOT/data/verl/val.parquet}"

export MASTERS_ROOT="$ROOT"
export PYTHONPATH="${ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export REWARD_BACKEND="${REWARD_BACKEND:-http}"
export REWARD_BASE_URL="${REWARD_BASE_URL:-http://127.0.0.1:8001/v1}"
export REWARD_MODEL="${REWARD_MODEL:-$MODEL_PATH}"
export RAG_PATH="${RAG_PATH:-$ROOT/data/train_data/rag.jsonl}"
export REWARD_AUDIT_DIR="${REWARD_AUDIT_DIR:-$ROOT/outputs/grpo_audit}"

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}

train_batch_size=${TRAIN_BATCH_SIZE:-4}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-4}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
ppo_micro_batch=${PPO_MICRO_BATCH:-1}

actor_lr=${ACTOR_LR:-5e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
lora_rank=${LORA_RANK:-16}
lora_alpha=${LORA_ALPHA:-32}
rollout_n=${ROLLOUT_N:-4}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
total_epochs=${TOTAL_EPOCHS:-1}
total_training_steps=${TOTAL_TRAINING_STEPS:-20}

project_name=${PROJECT_NAME:-masters_falsification_grpo}
experiment_name=${EXPERIMENT_NAME:-policy_lora_smoke}

cd "$VERL_ROOT"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="${train_batch_size}" \
  data.max_prompt_length="${max_prompt_length}" \
  data.max_response_length="${max_response_length}" \
  data.filter_overlong_prompts=True \
  data.truncation=right \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.lora_rank="${lora_rank}" \
  actor_rollout_ref.model.lora_alpha="${lora_alpha}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.optim.lr="${actor_lr}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ppo_micro_batch}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${rollout_gpu_mem_util}" \
  actor_rollout_ref.rollout.n="${rollout_n}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ppo_micro_batch}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${ppo_micro_batch}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.custom_reward_function.path="$ROOT/src/rl/falsification_reward.py" \
  reward.custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${experiment_name}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.val_before_train=False \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs="${total_epochs}" \
  trainer.total_training_steps="${total_training_steps}"
