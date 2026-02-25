@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

:: ============================================================
::  PaperBanana 论文图表助手 - Windows 一键启动脚本
::  双击此文件即可启动，首次运行自动安装所有依赖
:: ============================================================

:: --- 配置 ---
set "PYTHON_MIN_VER=3.10"
set "VENV_DIR=.venv"
set "RUNTIME_DIR=runtime"
set "PORT=8501"
set "APP_NAME=PaperBanana 论文图表助手"

:: --- 进入项目目录 ---
cd /d "%~dp0"

echo.
echo ==========================================
echo   %APP_NAME%
echo ==========================================
echo.

:: ============================================================
:: Step 1: 查找 / 安装 Python
:: ============================================================
set "PYTHON_CMD="

:: 1a. 检查 runtime\ 便携版 Python
if exist "%RUNTIME_DIR%\python\python.exe" (
    call :check_python_ver "%RUNTIME_DIR%\python\python.exe"
    if !errorlevel! == 0 (
        set "PYTHON_CMD=%RUNTIME_DIR%\python\python.exe"
        echo   [OK] 检测到便携版 Python
        goto :found_python
    )
)

:: 1b. 逐个检查系统 Python (避免 for 循环嵌套 errorlevel 问题)
call :try_system_python python  && goto :found_python
call :try_system_python python3 && goto :found_python
call :try_system_python py      && goto :found_python

:: 1c. 尝试 winget 安装 (Windows 10 1709+)
where winget >nul 2>&1 || goto :skip_winget
echo   [..] 使用 winget 安装 Python 3.12 ...
winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements >nul 2>&1 || goto :winget_failed
:: winget 安装后需要刷新 PATH
set "PATH=%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%PATH%"
call :try_system_python python && goto :found_python
:winget_failed
echo   [!!] winget 安装未成功，尝试下载便携版 ...
:skip_winget

:: 1d. 自动下载便携版 Python (python-build-standalone)
echo   [..] 未检测到 Python 3.10+，正在自动下载便携版 Python ...

if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"

:: 使用 PowerShell 查询 GitHub API 获取下载地址并下载
echo   [..] 查询最新版本 ...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ProgressPreference = 'SilentlyContinue'; " ^
    "try { " ^
    "  $releases = Invoke-RestMethod -Uri 'https://api.github.com/repos/indygreg/python-build-standalone/releases?per_page=5' -TimeoutSec 30; " ^
    "  $url = ''; " ^
    "  foreach ($r in $releases) { " ^
    "    foreach ($a in $r.assets) { " ^
    "      if ($a.name -match 'cpython-3\.12.*x86_64-pc-windows-msvc-install_only\.tar\.gz$') { " ^
    "        $url = $a.browser_download_url; break " ^
    "      } " ^
    "    }; " ^
    "    if ($url) { break } " ^
    "  }; " ^
    "  if (-not $url) { Write-Host '  [!!] 未找到下载地址'; exit 1 }; " ^
    "  Write-Host '  [..] 下载中 (约 40MB，请耐心等待) ...'; " ^
    "  $ProgressPreference = 'Continue'; " ^
    "  Invoke-WebRequest -Uri $url -OutFile '%RUNTIME_DIR%\python.tar.gz' -UseBasicParsing; " ^
    "  Write-Host '  [..] 解压中 ...'; " ^
    "  tar -xzf '%RUNTIME_DIR%\python.tar.gz' -C '%RUNTIME_DIR%\'; " ^
    "  Remove-Item '%RUNTIME_DIR%\python.tar.gz' -Force; " ^
    "  Write-Host '  [OK] 便携版 Python 已安装'; " ^
    "} catch { " ^
    "  Write-Host \"  [!!] 下载失败: $_\"; exit 1 " ^
    "}"

if exist "%RUNTIME_DIR%\python\python.exe" (
    set "PYTHON_CMD=%RUNTIME_DIR%\python\python.exe"
    goto :found_python
)

:: 所有方法都失败了
echo.
echo   [!!] 无法自动安装 Python，请手动安装:
echo.
echo       方法 1: Microsoft Store 搜索 "Python 3.12" 安装
echo       方法 2: 访问 https://www.python.org/downloads/ 下载安装
echo              安装时请勾选 "Add Python to PATH"
echo.
pause
exit /b 1

:found_python

:: ============================================================
:: Step 2: 创建 / 检查虚拟环境
:: ============================================================
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo   [..] 创建 Python 虚拟环境 ...
    "%PYTHON_CMD%" -m venv "%VENV_DIR%"
    if !errorlevel! neq 0 (
        echo   [!!] 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo   [OK] 虚拟环境已创建
) else (
    echo   [OK] 虚拟环境已存在
)

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"

:: ============================================================
:: Step 3: 安装 / 更新依赖
:: ============================================================
echo   [..] 检查并安装 Python 依赖 (首次较慢) ...
"%VENV_PIP%" install -r requirements.txt --quiet --disable-pip-version-check 2>nul
echo   [OK] 依赖已就绪

:: ============================================================
:: Step 4: 创建数据目录
:: ============================================================
if not exist "data\PaperBananaBench\diagram" mkdir "data\PaperBananaBench\diagram"
if not exist "data\PaperBananaBench\plot" mkdir "data\PaperBananaBench\plot"
if not exist "data\PaperBananaBench\diagram\ref.json" (
    >>"data\PaperBananaBench\diagram\ref.json" echo []
)
if not exist "data\PaperBananaBench\plot\ref.json" (
    >>"data\PaperBananaBench\plot\ref.json" echo []
)

:: ============================================================
:: Step 5: 清理残留端口 & 启动应用
:: ============================================================
echo   [..] 检查端口 %PORT% ...
for /f "tokens=5" %%A in ('netstat -ano 2^>nul ^| findstr ":%PORT% " ^| findstr "LISTENING"') do (
    echo   [!!] 端口 %PORT% 被占用 ^(PID: %%A^)，正在清理 ...
    taskkill /F /PID %%A >nul 2>&1
    timeout /t 1 /nobreak >nul
    echo   [OK] 端口已释放
)

echo.
echo ==========================================
echo   启动 %APP_NAME%
echo   浏览器将自动打开 http://localhost:%PORT%
echo   关闭此窗口可停止服务
echo ==========================================
echo.

:: 延迟打开浏览器
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:%PORT%"

:: 启动 Streamlit
"%VENV_DIR%\Scripts\streamlit.exe" run demo.py ^
    --server.port %PORT% ^
    --server.address 0.0.0.0 ^
    --server.headless true

pause
exit /b 0

:: ============================================================
:: 子程序: 检查 Python 版本 >= 3.10
:: ============================================================
:check_python_ver
%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>nul
exit /b !errorlevel!

:: ============================================================
:: 子程序: 尝试系统 Python
:: ============================================================
:try_system_python
where %~1 >nul 2>&1 || exit /b 1
%~1 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>nul || exit /b 1
set "PYTHON_CMD=%~1"
for /f "delims=" %%V in ('%~1 --version 2^>^&1') do echo   [OK] 检测到系统 %%V
exit /b 0
