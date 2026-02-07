视觉大模型数据中毒攻击研究工具（VisualModelPoisoningResearchTool）
================================================

> **重要免责声明**：本工具仅用于学术研究、安全测试和教育目的。使用者需遵守相关法律法规，不得用于任何非法或恶意活动。作者不对任何滥用行为承担任何责任。

本项目基于 BackdoorBox 框架，旨在为研究者提供一个可配置、可追踪、可复现的视觉大模型数据中毒攻击实验平台。

安装步骤
--------

```bash
# 1. 克隆 BackdoorBox 仓库到 external 目录
git clone https://github.com/THUYimingLi/BackdoorBox.git external/BackdoorBox

# 2. 安装依赖（建议使用虚拟环境）
pip install -r requirements.txt

# 3. 初始化配置（不会覆盖已存在配置）
python main.py config --init

# 4. 可选：验证配置
python main.py config --validate
```

可视化 Web 界面
----------------

如果你更倾向于图形化/可视化交互，可以启动内置的 Web 界面：

```bash
python run_web.py
```

然后在浏览器中访问 `http://127.0.0.1:8000` 即可进入可视化操作界面：

- 主页：功能入口与伦理警告
- “生成中毒数据集”页面：表单方式配置中毒参数并生成数据集
- “实验列表/详情”页面：查看和对比历史实验记录

使用示例
--------

### 基本数据集中毒

```bash
# 使用 BadNet 触发器生成中毒数据集
python main.py poison \
  --input-dir ./data/raw_datasets/cifar10 \
  --output-dir ./data/poisoned_datasets/cifar10_badnet \
  --trigger-type badnet \
  --poison-rate 0.1 \
  --target-label 0
```

### 使用攻击模板

```bash
# 使用预定义模板（如 stealthy_blend）
python main.py poison \
  --input-dir ./data/raw_datasets/imagenet \
  --output-dir ./data/poisoned_datasets/imagenet_stealthy \
  --template stealthy_blend
```

### 管理实验记录

```bash
# 查看所有实验
python main.py experiment --list

# 显示指定实验详情
python main.py experiment --show <experiment_uuid>

# 导出实验数据（导出为 JSON 文件）
python main.py experiment --export <experiment_uuid>
```

项目特性
--------

- **基于 BackdoorBox**：可对接 BadNet、Blend、ISSBA 等多种后门触发器。
- **纯PyTorch实现**：ISSBA功能使用BackdoorBox提供的纯PyTorch实现，无需TensorFlow。
- **可配置与可复现**：通过 `config/default_config.yaml` 与 `config/attack_templates.yaml` 管理攻击参数与模板。
- **实验记录完备**：使用 SQLite/SQLAlchemy 记录实验、攻击过程与模型评估结果，可导出为 JSON/CSV。
- **伦理与安全约束**：
  - 在主程序与 CLI 中打印伦理警告与使用限制。
  - 对中毒比例、数据集路径等进行安全检查，防止误用与误操作。

目录结构概览
------------

- `src/core/`：核心逻辑（攻击器、数据集与模型管理、触发器封装等）
- `src/database/`：实验记录数据库模型与操作
- `src/ui/`：命令行界面与配置管理
- `src/utils/`：文件工具、可视化、安全检查等
- `config/`：默认配置与攻击模板
- `data/`：原始数据集、中毒数据集与触发器资源



