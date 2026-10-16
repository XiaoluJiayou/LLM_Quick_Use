#!/usr/bin/env bash
# 依次运行出图、判定、汇总三步，日志同时写到 outputs/run_demo.log。
# 设置 PER_TYPE=k 时每个大类只跑前 k 条（冒烟测试），不设置则跑全量 227 条。
# JUDGE_MODE=fast|fast-slow；POLICY=策略路径或 default；WITH_PROMPT=1 时传提示词+图片。
# 每一步都能断点续跑，中断后重新运行本脚本即可。
set -eo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
# 输出经过 tee 写进日志，不关缓冲的话 tail -f 要隔很久才看到新内容
export PYTHONUNBUFFERED=1

mkdir -p outputs
exec > >(tee -a outputs/run_demo.log) 2>&1

args=()
if [ -n "$PER_TYPE" ]; then
    args=(--per-type "$PER_TYPE")
fi
mode_args=()
if [ -n "$JUDGE_MODE" ]; then
    mode_args=(--mode "$JUDGE_MODE")
fi
policy_args=()
if [ -n "$POLICY" ]; then
    policy_args=(--policy "$POLICY")
fi
prompt_args=()
if [ "${WITH_PROMPT:-0}" = "1" ]; then
    prompt_args=(--with-prompt)
fi

echo "===== $(date '+%F %T') 开始（PER_TYPE=${PER_TYPE:-全量}，JUDGE_MODE=${JUDGE_MODE:-fast}，POLICY=${POLICY:-默认文件}，WITH_PROMPT=${WITH_PROMPT:-0}）====="
python generate_images.py "${args[@]}"
python judge_singguard.py "${args[@]}" "${mode_args[@]}" "${policy_args[@]}" "${prompt_args[@]}"
python report.py "${args[@]}" "${mode_args[@]}" "${policy_args[@]}" "${prompt_args[@]}"
echo "===== $(date '+%F %T') 完成 ====="
