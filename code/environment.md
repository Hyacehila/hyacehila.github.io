# SSH相关内容文档
## NPM 环境安装 (NVM 方案)

在科研与开发环境中，为了避免权限问题并方便管理多版本，**强烈推荐使用 NVM (Node Version Manager)** 来安装 Node.js 和 NPM。

**本阶段一般使用人工完成**

**快速安装步骤：**

1.  **下载并安装 NVM**：
    ```bash
    # 方式一：使用 curl
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
    
    # 方式二：使用 wget (如果系统中没有 curl)
    wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
    ```

2.  **激活环境**：
    执行以下命令重载配置（或直接重启终端）：
    ```bash
    source ~/.bashrc  # 若使用 Zsh，请替换为 ~/.zshrc
    ```

3.  **安装 LTS 版本**：
    安装长期支持版 Node.js（会自动包含 NPM）：
    ```bash
    nvm install --lts
    ```

4.  **验证安装**：
    ```bash
    node -v
    npm -v
    ```

5.  **安装 Codex CLI**：
    ```bash
    npm i -g @openai/codex
    ```
6.  **Claude Code**:
    ```bash
    curl -fsSL https://claude.ai/install.sh | bash
    ```

## SSH服务器环境控制

首先需要在服务器上安装 uv

```bash
#在shh服务器上安装uv,两个方案任选其一
wget -qO- https://astral.sh/uv/install.sh | sh

curl -LsSf https://astral.sh/uv/install.sh | sh
```

初始化项目，指定新的Python版本（根据项目需求确定），下载Python，确认我们进入了uv的虚拟环境，如果是打开一个已经初始化过的项目，无需重新初始化项目，直接检查相关环境。

```bash
uv init  
uv python pin 3.12
uv python install 3.12
uv run which python 
```

手动修改 `toml` 实现对torch环境的手动控制，根据情况决定

```python
dependencies = [
    "torch",
    "torchvision",
    "bitsandbytes",
    "datasets",
    "peft",
    "transformers",
    "trl",
]

# --- 关键配置开始 ---

# 定义一个名为 pytorch 的额外源
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu121"
explicit = true  # 设为 true，表示这个源专门用于指定的包，不混用

# 告诉 uv，只有 torch 和 torchvision 去上面定义的 'pytorch' 源找
# 其他所有包（比如 markupsafe, numpy）还是去默认的 PyPI 下载
[tool.uv.sources]
torch = { index = "pytorch" }
torchvision = { index = "pytorch" }

# 将默认源修改为 tsinghua 的镜像
[tool.uv]
index-url = "https://pypi.tuna.tsinghua.edu.cn/simple/"

# --- 关键配置结束 ---
```

根据toml文件重建环境，由于torch的巨大体积以及镜像源的问题，重建速度一般会很慢，如果以前安装的环境被移除了，使用 sync 命令从toml文件重建环境。

```bash
uv sync
```

根据需求，后面的包可以使用 uv add 添加进入toml 也可以使用 uv pip install 恢复原始的管理方法





## SSH公私钥设计

检查本地公钥私钥
```bash
Get-ChildItem -Path $env:USERPROFILE\.ssh -Force
```

如果没有，可以新建一个，酌情设置passphrase
```bash
ssh-keygen -t ed25519 -C "your_email@example.com" 
```

列出本地公钥并复制其中内容，以下是我真实的本地公钥，如果位于云服务器环境则需要将本地公钥添加到SSH的服务器中

```bash
cat ~/.ssh/id_ed25519.pub

ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIL0BPxmDPY0oowpSqH2RK5t4txCqJfmU0y4cmbHwyc9b hyacehila@gmail.com
```

登录到远程服务器

```bash
ssh username@hostname
```

依次执行

```bash
# 创建 .ssh 目录（如果不存在）
mkdir -p ~/.ssh

# 将你复制的公钥内容追加到 authorized_keys 文件中
# 将 "PASTE_YOUR_PUBLIC_KEY_HERE" 替换成你刚刚复制的内容
echo ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIL0BPxmDPY0oowpSqH2RK5t4txCqJfmU0y4cmbHwyc9b hyacehila@gmail.com  >> ~/.ssh/authorized_keys

# 设置正确的权限（这一步至关重要！）
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```


## SSH的终端控制 Zellij

[Zellij官方文档](https://zellij.dev/documentation/integration.html)

这样就允许我们在Linux服务器上进行多终端的控制，以及避免本地关闭导致远程终端也被关闭。


## 第三方训练情况监控 SwanLab
Swanlab 替代了 wanlab的功能，更加方便在国内访问