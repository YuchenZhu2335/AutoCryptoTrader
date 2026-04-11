# ============================================
# AutoCryptoTrader - Windows 开发环境一键配置脚本
# 使用方法: 右键此文件 -> "使用 PowerShell 运行"
# ============================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Windows 开发环境一键配置脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# --- 检测是否有管理员权限 ---
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "[!] 需要管理员权限，正在提权重启..." -ForegroundColor Yellow
    Start-Process powershell.exe "-ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    exit
}

# --- 检测并安装 winget (Windows 包管理器) ---
function Test-Winget {
    try {
        $null = Get-Command winget -ErrorAction Stop
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-Winget)) {
    Write-Host "[!] winget 未找到，请先从 Microsoft Store 安装 'App Installer'" -ForegroundColor Red
    Write-Host "    或访问: https://aka.ms/getwinget" -ForegroundColor Yellow
    Read-Host "安装完 winget 后按 Enter 继续..."
}

# --- 辅助函数 ---
function Install-IfMissing {
    param(
        [string]$Command,
        [string]$DisplayName,
        [string]$WingetId,
        [string]$VerifyCmd = "$Command --version"
    )

    Write-Host ""
    Write-Host "--- 检查 $DisplayName ---" -ForegroundColor Yellow

    try {
        $null = Get-Command $Command -ErrorAction Stop
        $ver = Invoke-Expression $VerifyCmd 2>&1 | Select-Object -First 1
        Write-Host "[OK] $DisplayName 已安装: $ver" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "[..] $DisplayName 未安装，正在安装..." -ForegroundColor Cyan
        winget install --id $WingetId --accept-source-agreements --accept-package-agreements --silent
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] $DisplayName 安装成功！" -ForegroundColor Green
            return $true
        } else {
            Write-Host "[FAIL] $DisplayName 安装失败，请手动安装" -ForegroundColor Red
            return $false
        }
    }
}

# ============================================
# 1. 安装核心工具
# ============================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  第1步: 安装核心开发工具" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Node.js LTS (Claude Code 依赖)
Install-IfMissing -Command "node" -DisplayName "Node.js" -WingetId "OpenJS.NodeJS.LTS"

# Git
Install-IfMissing -Command "git" -DisplayName "Git" -WingetId "Git.Git"

# Python 3.11
Install-IfMissing -Command "python" -DisplayName "Python" -WingetId "Python.Python.3.11"

# ============================================
# 2. 刷新 PATH (让新安装的工具生效)
# ============================================
Write-Host ""
Write-Host "--- 刷新环境变量 ---" -ForegroundColor Yellow
$env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")

# ============================================
# 3. 配置国内镜像加速
# ============================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  第2步: 配置国内镜像加速" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# pip 换清华源
Write-Host "--- 配置 pip 清华镜像源 ---" -ForegroundColor Yellow
try {
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | Out-Null
    Write-Host "[OK] pip 已切换到清华镜像源" -ForegroundColor Green
} catch {
    Write-Host "[SKIP] pip 镜像配置跳过" -ForegroundColor Yellow
}

# npm 换淘宝源
Write-Host "--- 配置 npm 淘宝镜像源 ---" -ForegroundColor Yellow
try {
    npm config set registry https://registry.npmmirror.com 2>&1 | Out-Null
    Write-Host "[OK] npm 已切换到淘宝镜像源" -ForegroundColor Green
} catch {
    Write-Host "[SKIP] npm 镜像配置跳过" -ForegroundColor Yellow
}

# ============================================
# 4. 安装 Claude Code CLI
# ============================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  第3步: 安装 Claude Code" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

Write-Host "--- 安装 Claude Code CLI ---" -ForegroundColor Yellow
try {
    npm install -g @anthropic-ai/claude-code 2>&1
    Write-Host "[OK] Claude Code CLI 安装成功！" -ForegroundColor Green
} catch {
    Write-Host "[FAIL] Claude Code CLI 安装失败，请稍后手动运行: npm install -g @anthropic-ai/claude-code" -ForegroundColor Red
}

# ============================================
# 5. 安装 Claude Desktop (桌面应用)
# ============================================
Write-Host ""
Write-Host "--- 安装 Claude Desktop ---" -ForegroundColor Yellow
winget install --id Anthropic.Claude --accept-source-agreements --accept-package-agreements --silent 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Claude Desktop 安装成功！" -ForegroundColor Green
} else {
    Write-Host "[INFO] Claude Desktop 可能需要手动下载: https://claude.ai/download" -ForegroundColor Yellow
}

# ============================================
# 6. 克隆项目仓库并配置
# ============================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  第4步: 克隆项目仓库" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$projectDir = "$HOME\AutoCryptoTrader"

if (Test-Path $projectDir) {
    Write-Host "[OK] 项目已存在: $projectDir" -ForegroundColor Green
} else {
    Write-Host "--- 克隆 AutoCryptoTrader 仓库 ---" -ForegroundColor Yellow
    git clone https://github.com/YuchenZhu2335/AutoCryptoTrader.git $projectDir 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] 仓库克隆成功！" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] 克隆失败，可能需要先登录 GitHub" -ForegroundColor Red
    }
}

# ============================================
# 7. 创建 Python 虚拟环境并安装依赖
# ============================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  第5步: 配置 Python 项目环境" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (Test-Path $projectDir) {
    Set-Location $projectDir

    if (-not (Test-Path "venv")) {
        Write-Host "--- 创建 Python 虚拟环境 ---" -ForegroundColor Yellow
        python -m venv venv
    }

    Write-Host "--- 激活虚拟环境并安装依赖 ---" -ForegroundColor Yellow
    & "$projectDir\venv\Scripts\Activate.ps1"
    pip install --upgrade pip setuptools wheel 2>&1 | Out-Null
    pip install -r requirements.txt 2>&1
    Write-Host "[OK] Python 依赖安装完成！" -ForegroundColor Green
}

# ============================================
# 最终报告
# ============================================
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  配置完成！环境检查报告:" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

$tools = @(
    @{Name="Node.js"; Cmd="node --version"},
    @{Name="npm"; Cmd="npm --version"},
    @{Name="Git"; Cmd="git --version"},
    @{Name="Python"; Cmd="python --version"},
    @{Name="pip"; Cmd="pip --version"},
    @{Name="Claude Code"; Cmd="claude --version"}
)

foreach ($tool in $tools) {
    try {
        $ver = Invoke-Expression $tool.Cmd 2>&1 | Select-Object -First 1
        Write-Host "  [OK] $($tool.Name): $ver" -ForegroundColor Green
    } catch {
        Write-Host "  [X]  $($tool.Name): 未安装" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  下一步操作:" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  1. 打开 Claude Desktop 并登录你的账号" -ForegroundColor White
Write-Host "  2. 或在终端输入 'claude' 启动 Claude Code CLI" -ForegroundColor White
Write-Host "  3. 项目目录: $projectDir" -ForegroundColor White
Write-Host ""

Read-Host "按 Enter 退出"
