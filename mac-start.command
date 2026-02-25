#!/bin/bash
# ============================================================
#  PaperBanana 论文图表助手 — macOS 一键启动脚本
#  双击此文件即可启动，首次运行自动安装所有依赖
# ============================================================

set -euo pipefail

# ─── 配置 ──────────────────────────────────────────────────
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10
VENV_DIR=".venv"
RUNTIME_DIR="runtime"
PORT=8501
APP_NAME="PaperBanana 论文图表助手"

# ─── 进入项目目录（.command 文件从 Finder 打开时 cwd 是 $HOME）──
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

# ─── 颜色与输出 ────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

banner()  { echo -e "\n${CYAN}${BOLD}$1${NC}"; }
step()    { echo -e "  ${GREEN}▸${NC} $1"; }
ok()      { echo -e "  ${GREEN}✔${NC} $1"; }
warn()    { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()    { echo -e "  ${RED}✖${NC} $1"; }

# ─── 辅助：检查 Python 版本 >= 3.10 ────────────────────────
check_py_ver() {
    local py="$1"
    "$py" -c "
import sys
if sys.version_info >= ($PYTHON_MIN_MAJOR, $PYTHON_MIN_MINOR):
    sys.exit(0)
else:
    sys.exit(1)
" 2>/dev/null
}

# ============================================================
banner "=========================================="
banner "  $APP_NAME"
banner "=========================================="
echo ""

# ─── Step 1: 查找 / 安装 Python ────────────────────────────
PYTHON_CMD=""

# 1a. 已有 runtime/ 中的便携版 Python
if [ -x "$RUNTIME_DIR/python/bin/python3" ]; then
    if check_py_ver "$RUNTIME_DIR/python/bin/python3"; then
        PYTHON_CMD="$RUNTIME_DIR/python/bin/python3"
        ok "检测到便携版 Python: $PYTHON_CMD"
    fi
fi

# 1b. 系统 python3
if [ -z "$PYTHON_CMD" ]; then
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null && check_py_ver "$cmd"; then
            PYTHON_CMD="$cmd"
            ok "检测到系统 Python: $($cmd --version 2>&1)"
            break
        fi
    done
fi

# 1c. 尝试 Homebrew 安装
if [ -z "$PYTHON_CMD" ]; then
    if command -v brew &>/dev/null; then
        step "使用 Homebrew 安装 Python 3.12 ..."
        brew install python@3.12
        PYTHON_CMD="$(brew --prefix python@3.12)/bin/python3"
        ok "Homebrew 安装完成"
    fi
fi

# 1d. 自动下载便携版 Python (python-build-standalone)
if [ -z "$PYTHON_CMD" ]; then
    step "未检测到 Python ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}+，正在自动下载便携版 Python ..."
    ARCH="$(uname -m)"

    # 通过 GitHub API 获取最新的 python-build-standalone 下载地址
    if [ "$ARCH" = "arm64" ]; then
        PATTERN="cpython-3.12.*aarch64-apple-darwin-install_only.tar.gz"
    else
        PATTERN="cpython-3.12.*x86_64-apple-darwin-install_only.tar.gz"
    fi

    step "查询最新版本 ..."
    DOWNLOAD_URL=$(curl -sL "https://api.github.com/repos/indygreg/python-build-standalone/releases?per_page=5" \
        | grep -o "https://[^\"]*${PATTERN}" | head -1 || true)

    if [ -z "$DOWNLOAD_URL" ]; then
        fail "无法自动获取 Python 下载地址。"
        echo ""
        echo "  请手动安装 Python 3.10+："
        echo "    方法 1: 安装 Homebrew 后重试  →  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "    方法 2: 从 https://www.python.org/downloads/ 下载安装"
        echo ""
        read -n 1 -s -r -p "按任意键退出 ..."
        exit 1
    fi

    step "下载中 (约 50MB，请耐心等待) ..."
    mkdir -p "$RUNTIME_DIR"
    curl -L --progress-bar -o "$RUNTIME_DIR/python.tar.gz" "$DOWNLOAD_URL"

    step "解压中 ..."
    tar -xzf "$RUNTIME_DIR/python.tar.gz" -C "$RUNTIME_DIR/"
    rm -f "$RUNTIME_DIR/python.tar.gz"

    PYTHON_CMD="$RUNTIME_DIR/python/bin/python3"
    if [ -x "$PYTHON_CMD" ] && check_py_ver "$PYTHON_CMD"; then
        ok "便携版 Python 已安装到 $RUNTIME_DIR/python/"
    else
        fail "下载的 Python 无法运行，请手动安装 Python 3.10+"
        read -n 1 -s -r -p "按任意键退出 ..."
        exit 1
    fi
fi

# ─── Step 2: 创建 / 检查虚拟环境 ───────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    step "创建 Python 虚拟环境 ..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    ok "虚拟环境已创建: $VENV_DIR/"
else
    ok "虚拟环境已存在: $VENV_DIR/"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# ─── Step 3: 安装 / 更新依赖 ───────────────────────────────
step "检查并安装 Python 依赖 (首次较慢) ..."
"$VENV_PIP" install -r requirements.txt --quiet --disable-pip-version-check 2>&1 | tail -1 || true
ok "依赖已就绪"

# ─── Step 4: 创建数据目录 ──────────────────────────────────
mkdir -p data/PaperBananaBench/diagram data/PaperBananaBench/plot
[ -f data/PaperBananaBench/diagram/ref.json ] || echo "[]" > data/PaperBananaBench/diagram/ref.json
[ -f data/PaperBananaBench/plot/ref.json ]    || echo "[]" > data/PaperBananaBench/plot/ref.json

# ─── Step 5: 清理残留端口 & 启动应用 ─────────────────────────
OLD_PID=$(lsof -ti:"$PORT" 2>/dev/null || true)
if [ -n "$OLD_PID" ]; then
    warn "端口 $PORT 被占用 (PID: $OLD_PID)，正在清理 ..."
    kill -9 $OLD_PID 2>/dev/null || true
    sleep 1
    ok "端口已释放"
fi

echo ""
banner "=========================================="
banner "  启动 $APP_NAME"
banner "  浏览器将自动打开 http://localhost:$PORT"
banner "  按 Ctrl+C 可停止服务"
banner "=========================================="
echo ""

# 延迟打开浏览器
(sleep 3 && open "http://localhost:$PORT") &

# 启动 Streamlit
"$VENV_DIR/bin/streamlit" run demo.py \
    --server.port "$PORT" \
    --server.address 0.0.0.0 \
    --server.headless true
