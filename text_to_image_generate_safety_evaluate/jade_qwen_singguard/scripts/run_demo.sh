#!/usr/bin/env bash
# 依次运行出图、判定、汇总三步，日志同时写到 outputs/run_demo.log。
# 设置 PER_TYPE=k 时每个数据集大类只跑前 k 条，不设置则跑 CSV 全量。
# JUDGE_MODE=fast|fast-slow；POLICY=策略路径或 default；WITH_PROMPT=1 时传提示词+图片。
# PYTHON_BIN 可选择 Python 解释器；DRY_RUN=1 只检查数据和任务，不加载模型。
# 任一步失败会以非零状态停止；重新运行可续跑与当前配置匹配的已完成项。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# 输出经过 tee 写进日志，不关缓冲的话 tail -f 要隔很久才看到新内容
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
WITH_PROMPT="${WITH_PROMPT:-0}"
if [[ "$DRY_RUN" != "0" && "$DRY_RUN" != "1" ]]; then
    echo "DRY_RUN 只能是 0 或 1" >&2
    exit 2
fi
if [[ "$WITH_PROMPT" != "0" && "$WITH_PROMPT" != "1" ]]; then
    echo "WITH_PROMPT 只能是 0 或 1" >&2
    exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "找不到 Python 解释器: $PYTHON_BIN" >&2
    exit 2
fi

LOG_FILE="${LOG_FILE:-outputs/run_demo.log}"
mkdir -p -- "$(dirname -- "$LOG_FILE")"
exec > >(tee -a -- "$LOG_FILE") 2>&1

args=()
if [ -n "${PER_TYPE:-}" ]; then
    args=(--per-type "$PER_TYPE")
fi
mode_args=()
if [ -n "${JUDGE_MODE:-}" ]; then
    mode_args=(--mode "$JUDGE_MODE")
fi
policy_args=()
if [ -n "${POLICY:-}" ]; then
    policy_args=(--policy "$POLICY")
fi
prompt_args=()
if [ "${WITH_PROMPT:-0}" = "1" ]; then
    prompt_args=(--with-prompt)
fi

dry_args=()
if [ "$DRY_RUN" = "1" ]; then
    dry_args=(--dry-run)
fi

echo "===== $(date '+%F %T') 开始（PER_TYPE=${PER_TYPE:-全量}，JUDGE_MODE=${JUDGE_MODE:-fast}，POLICY=${POLICY:-默认文件}，WITH_PROMPT=${WITH_PROMPT:-0}，DRY_RUN=$DRY_RUN）====="
"$PYTHON_BIN" generate_images.py "${args[@]}" "${dry_args[@]}"
"$PYTHON_BIN" judge_singguard.py "${args[@]}" "${mode_args[@]}" "${policy_args[@]}" "${prompt_args[@]}" "${dry_args[@]}"
if [ "$DRY_RUN" != "1" ]; then
    "$PYTHON_BIN" report.py "${args[@]}" "${mode_args[@]}" "${policy_args[@]}" "${prompt_args[@]}"
fi
echo "===== $(date '+%F %T') 完成 ====="
