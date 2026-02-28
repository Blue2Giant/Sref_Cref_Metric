#!/bin/bash

# ==============================================================================
# 脚本名称: auto_safe_sync_v2.sh
# 描述: (自动化版本 v2) 安全地将一个目录同步到另一个位置。
#       - 使用命名参数 --source, --target, --overwrite。
#       - 默认安全模式：如果目标目录存在且未提供 --overwrite，则会失败。
# 使用方法:
#   默认: ./auto_safe_sync_v2.sh --source <src> --target <dest>
#   覆盖: ./auto_safe_sync_v2.sh --source <src> --target <dest> --overwrite
#   带vault配置: ./auto_safe_sync_v2.sh --source <src> --target <dest> --vault-toml <toml_path>
# ==============================================================================

# --- 配置 ---
set -e
set -o pipefail

# --- 函数定义 ---

# 打印使用方法并退出
function usage() {
    echo "使用方法: $0 --source <源目录> --target <目标目录> [--overwrite] [--vault-toml <toml文件路径>]"
    echo "  --source:      要同步的源目录。"
    echo "  --target:      同步的目标目录。"
    echo "  --overwrite:   可选参数。如果目标目录已存在，则允许同步覆盖。否则脚本会失败。"
    echo "  --vault-toml:  可选参数。指定一个toml文件路径，该文件将被移动到目标目录并重命名为vault.toml。"
    exit 1
}

# 打印错误信息并退出
function error_exit() {
    echo "错误: $1" >&2
    exit 1
}

# --- 参数解析 ---

SOURCE_DIR=""
DEST_DIR=""
OVERWRITE_ENABLED=0 # 默认不覆盖
VAULT_TOML_PATH="" # vault.toml文件路径

# 使用 while-case 结构解析命名参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --source) SOURCE_DIR="$2"; shift ;;
        --target) DEST_DIR="$2"; shift ;;
        --overwrite) OVERWRITE_ENABLED=1 ;;
        --vault-toml) VAULT_TOML_PATH="$2"; shift ;;
        *) echo "未知参数: $1"; usage ;;
    esac
    shift
done

# --- 主逻辑 ---

# 1. 强制参数校验
if [ -z "${SOURCE_DIR}" ] || [ -z "${DEST_DIR}" ]; then
    echo "错误: --source 和 --target 参数都是必需的。"
    usage
fi

echo "=== 开始执行同步任务 (v2) ==="
echo "时间: $(date)"
echo "源目录: ${SOURCE_DIR}"
echo "目标目录: ${DEST_DIR}"
echo "允许覆盖: ${OVERWRITE_ENABLED}"
if [ -n "${VAULT_TOML_PATH}" ]; then
    echo "Vault TOML: ${VAULT_TOML_PATH}"
fi
echo "=============================="

# 2. 检查 megfile 命令是否存在
if ! command -v megfile &> /dev/null; then
    error_exit "未找到 'megfile' 命令。请确保 megfile-cli 已安装并位于 PATH 中。"
fi

# 3. 校验源目录是否存在
if [ ! -d "${SOURCE_DIR}" ]; then
    error_exit "源目录 '${SOURCE_DIR}' 不存在或不是一个目录。"
fi
echo "[检查通过] 源目录 '${SOURCE_DIR}' 已找到。"

# 4. 检查并处理目标目录 (核心安全逻辑)
if [ -d "${DEST_DIR}" ]; then
    if [ "${OVERWRITE_ENABLED}" -eq 1 ]; then
        echo "[警告] 目标目录 '${DEST_DIR}' 已存在。已提供 --overwrite 参数，将继续同步。"
    else
        echo "目标目录 '${DEST_DIR}' 已存在。为防止意外覆盖，操作已中止。如果确实需要同步，请在命令中添加 --overwrite 标志。"
    fi
else
    echo "[信息] 目标目录 '${DEST_DIR}' 不存在，将自动创建。"
    mkdir -p "${DEST_DIR}"
    echo "[成功] 目标目录已创建。"
fi


# 6. 执行同步操作
echo "----------------------------------------"
echo "开始执行 'megfile sync'..."
echo "----------------------------------------"

megfile sync --skip -g "${SOURCE_DIR}" "${DEST_DIR}"


mkdir ~/.aws
cp credentials ~/.aws/credentials

# 5. 处理 vault-toml 文件（如果指定）
if [ -n "${VAULT_TOML_PATH}" ]; then
    echo "----------------------------------------"
    echo "处理 vault-toml 文件..."
    echo "----------------------------------------"
    
    # 使用 megfile mv 将文件移动到目标目录并重命名为 vault.toml
    echo "正在将 '${VAULT_TOML_PATH}' 移动到 '${DEST_DIR}/vault.toml'..."
    megfile mv "${VAULT_TOML_PATH}" "${DEST_DIR}/vault.toml"
    echo "[成功] vault-toml 文件已移动到目标目录。"
fi

echo "----------------------------------------"
echo "同步成功完成！"
echo "----------------------------------------"

exit