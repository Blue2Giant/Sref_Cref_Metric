# Vault项目Makefile使用指南

这个Makefile为Vault项目提供了核心的开发、测试和部署工具链，基于`uv`包管理器构建。专注于最常用的功能，依赖管理建议直接使用`uv`命令。

## 🚀 快速开始

### 1. 查看所有可用命令
```bash
make help
```

### 2. 开发环境设置
```bash
make dev-setup
```

### 3. 运行测试
```bash
make test
```

### 4. 启动Gradio Demo
```bash
make demo
```

## 📦 包管理命令

### 依赖管理
```bash
# 安装所有依赖（包括开发依赖）
make install

# 仅安装生产环境依赖
make install-prod

# 同步依赖到最新版本
make sync

# 更新锁文件
make lock

# 显示依赖树
make show-deps
```

### 依赖管理（建议直接使用uv）
```bash
# 添加新依赖
uv add requests

# 添加开发依赖
uv add --dev pytest-cov

# 移除依赖
uv remove requests

# 查看依赖树
make show-deps
```

## 🧪 测试命令

### 基本测试
```bash
# 运行所有测试
make test

# 运行测试（详细输出）
make test-verbose

# 运行测试并生成覆盖率报告
make test-coverage

# 监视文件变化并自动运行测试
make test-watch
```

### 特定测试
```bash
# 运行特定测试文件
make test-specific FILE=test_lance_taker.py

# 快速测试（清理 + 测试）
make quick-test
```

## 🎨 Gradio Demo命令

### 启动Demo
```bash
# 启动Gradio demo（默认端口7860）
make demo
```

## 🔧 代码质量命令

### 代码格式化
```bash
# 格式化代码
make format

# 运行代码检查
make lint

# 运行完整的代码检查（格式化 + 检查 + 测试）
make check
```

## 🧹 清理命令

### 基本清理
```bash
# 清理临时文件和缓存
make clean

# 深度清理（包括依赖）
make clean-all
```

## 🏗️ 构建和发布

### 构建
```bash
# 构建项目包
make build

# 执行完整的构建流程
make all
```

### 发布
```bash
# 发布包到PyPI
make publish

# 发布包到测试PyPI
make publish-test
```

## 🛠️ 开发工具

### 环境管理
```bash

# 显示项目信息
make info

# 显示依赖信息
make deps
```

### 开发环境
```bash
# 启动项目虚拟环境shell
make shell
```

## 📋 常用工作流

### 1. 新功能开发
```bash
make dev-setup    # 设置开发环境
make test         # 运行测试确保基础功能正常
# ... 开发代码 ...
make format       # 格式化代码
make lint         # 检查代码质量
make test         # 运行测试
```

### 2. 提交前检查
```bash
make check        # 运行完整的代码检查
```

### 3. Demo演示
```bash
make demo         # 启动demo进行演示
```

### 4. 发布准备
```bash
make clean        # 清理临时文件
make test         # 运行测试
make build        # 构建包
```

## 🎯 特殊功能

### 颜色输出
Makefile使用颜色输出，让命令执行结果更加清晰易读：
- 🔵 蓝色：正在执行的操作
- 🟢 绿色：成功完成的操作
- 🟡 黄色：提示信息
- 🔴 红色：错误信息

### 错误处理
所有命令都包含适当的错误处理和用户友好的错误消息。

### 参数化命令
某些命令支持参数化使用，如：
- `make test-specific FILE=test_file.py`

## 📝 注意事项

1. **虚拟环境**：所有Python命令都通过`uv run`在项目虚拟环境中执行
2. **依赖管理**：使用`uv`作为包管理器，建议直接使用`uv add/remove`命令管理依赖
3. **测试环境**：测试命令会自动发现`tests/`目录下的所有测试文件
4. **Demo路径**：Gradio demo默认启动`examples/apps/vault_browser.py`
5. **清理安全**：清理命令会删除临时文件，但不会影响源代码

## 🔗 相关文件

- `pyproject.toml` - 项目配置和依赖定义
- `uv.lock` - 依赖锁定文件
- `examples/apps/vault_browser.py` - Gradio demo应用
- `tests/` - 测试文件目录

---

如有问题或建议，请查看项目文档或联系开发团队。
