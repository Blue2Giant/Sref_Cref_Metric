# Vault项目Makefile
# 提供常用的开发、测试和部署命令

# 默认目标
.DEFAULT_GOAL := help

# 项目配置
PROJECT_NAME := vault
PYTHON := python3
UV := uv
PYTEST := pytest
GRADIO_DEMO := examples/apps/vault_browser.py

# 颜色定义
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m

# 帮助信息
.PHONY: help
help: ## 显示此帮助信息
	@echo "$(CYAN)Vault项目Makefile$(RESET)"
	@echo "$(CYAN)==================$(RESET)"
	@echo ""
	@echo "$(YELLOW)可用命令:$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)示例用法:$(RESET)"
	@echo "  make install     # 安装项目依赖"
	@echo "  make test        # 运行测试"
	@echo "  make demo        # 启动Gradio demo"
	@echo "  make build-sync  # 构建并同步到S3"
	@echo "  make clean       # 清理临时文件"
	@echo ""
	@echo "$(YELLOW)依赖管理建议直接使用uv:$(RESET)"
	@echo "  uv add package_name     # 添加依赖"
	@echo "  uv remove package_name  # 移除依赖"

# =============================================================================
# UV包管理相关命令
# =============================================================================

.PHONY: install
install: ## 安装项目依赖（包括开发依赖）
	@echo "$(BLUE)安装项目依赖...$(RESET)"
	$(UV) sync --all-extras
	@echo "$(GREEN)依赖安装完成!$(RESET)"

.PHONY: install-prod
install-prod: ## 仅安装生产环境依赖
	@echo "$(BLUE)安装生产环境依赖...$(RESET)"
	$(UV) sync --no-dev
	@echo "$(GREEN)生产环境依赖安装完成!$(RESET)"

.PHONY: sync
sync: ## 同步依赖到最新版本
	@echo "$(BLUE)同步依赖...$(RESET)"
	$(UV) sync --upgrade
	@echo "$(GREEN)依赖同步完成!$(RESET)"

.PHONY: lock
lock: ## 更新锁文件
	@echo "$(BLUE)更新锁文件...$(RESET)"
	$(UV) lock
	@echo "$(GREEN)锁文件更新完成!$(RESET)"


.PHONY: show-deps
show-deps: ## 显示项目依赖树
	@echo "$(BLUE)项目依赖树:$(RESET)"
	$(UV) tree

# =============================================================================
# 测试相关命令
# =============================================================================

.PHONY: test
test: ## 运行所有测试
	@echo "$(BLUE)运行测试...$(RESET)"
	$(UV) run $(PYTEST) tests/ -v --tb=short
	@echo "$(GREEN)测试完成!$(RESET)"

.PHONY: test-verbose
test-verbose: ## 运行测试（详细输出）
	@echo "$(BLUE)运行测试（详细模式）...$(RESET)"
	$(UV) run $(PYTEST) tests/ -v -s --tb=long
	@echo "$(GREEN)测试完成!$(RESET)"

.PHONY: test-coverage
test-coverage: ## 运行测试并生成覆盖率报告
	@echo "$(BLUE)运行测试并生成覆盖率报告...$(RESET)"
	$(UV) run $(PYTEST) tests/ --cov=src/$(PROJECT_NAME) --cov-report=html --cov-report=term
	@echo "$(GREEN)覆盖率报告已生成: htmlcov/index.html$(RESET)"

.PHONY: test-watch
test-watch: ## 监视文件变化并自动运行测试
	@echo "$(BLUE)启动测试监视模式...$(RESET)"
	$(UV) run $(PYTEST) tests/ -f --tb=short

.PHONY: test-specific
test-specific: ## 运行特定测试文件 (用法: make test-specific FILE=test_file.py)
	@if [ -z "$(FILE)" ]; then \
		echo "$(RED)错误: 请指定测试文件$(RESET)"; \
		echo "$(YELLOW)用法: make test-specific FILE=test_file.py$(RESET)"; \
		exit 1; \
	fi
	@echo "$(BLUE)运行测试文件: $(FILE)$(RESET)"
	$(UV) run $(PYTEST) tests/$(FILE) -v --tb=short

# =============================================================================
# Gradio Demo相关命令
# =============================================================================

.PHONY: demo
demo: ## 启动Gradio demo (vault_browser.py)
	@echo "$(BLUE)启动Gradio demo...$(RESET)"
	@echo "$(YELLOW)Demo将在 http://localhost:7860 启动$(RESET)"
	@echo "$(YELLOW)按 Ctrl+C 停止服务$(RESET)"
	$(UV) run $(PYTHON) $(GRADIO_DEMO)

# =============================================================================
# 代码质量相关命令
# =============================================================================

.PHONY: format
format: ## 格式化代码
	@echo "$(BLUE)格式化代码...$(RESET)"
	$(UV) run black src/ tests/ examples/
	$(UV) run isort src/ tests/ examples/
	@echo "$(GREEN)代码格式化完成!$(RESET)"

.PHONY: lint
lint: ## 运行代码检查
	@echo "$(BLUE)运行代码检查...$(RESET)"
	$(UV) run flake8 src/ tests/ examples/
	$(UV) run mypy src/
	@echo "$(GREEN)代码检查完成!$(RESET)"

.PHONY: check
check: format lint test ## 运行完整的代码检查（格式化 + 检查 + 测试）
	@echo "$(GREEN)所有检查完成!$(RESET)"

# =============================================================================
# 构建和发布相关命令
# =============================================================================

.PHONY: build
build: ## 构建项目包
	@echo "$(BLUE)构建项目包...$(RESET)"
	$(UV) build
	@echo "$(GREEN)项目包构建完成!$(RESET)"

.PHONY: sync-s3
sync-s3: build ## 同步构建包到S3
	@echo "$(BLUE)同步构建包到S3...$(RESET)"
	megfile sync -g ./dist s3://ruiwang/pypi/vault/
	@echo "$(BLUE)生成最新版本信息...$(RESET)"
	@LATEST=$$(ls dist/step_vault-*.whl | sort -V | tail -n 1 | xargs -n 1 basename); \
	echo $$LATEST > dist/latest.txt; \
	megfile cp dist/latest.txt s3://ruiwang/pypi/vault/latest.txt
	@echo "$(GREEN)S3同步完成! 最新版本: $$(cat dist/latest.txt)$(RESET)"
	
.PHONY: build-sync
build-sync: build sync-s3 ## 构建项目包并同步到S3
	@echo "$(GREEN)构建和S3同步完成!$(RESET)"

.PHONY: publish
publish: ## 发布包到PyPI
	@echo "$(BLUE)发布包到PyPI...$(RESET)"
	$(UV) publish
	@echo "$(GREEN)包发布完成!$(RESET)"

.PHONY: publish-test
publish-test: ## 发布包到测试PyPI
	@echo "$(BLUE)发布包到测试PyPI...$(RESET)"
	$(UV) publish --index-url https://test.pypi.org/simple/
	@echo "$(GREEN)包发布到测试PyPI完成!$(RESET)"

# =============================================================================
# 清理相关命令
# =============================================================================

.PHONY: clean
clean: ## 清理临时文件和缓存
	@echo "$(BLUE)清理临时文件...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .tox/ .coverage htmlcov/
	@echo "$(GREEN)清理完成!$(RESET)"

.PHONY: clean-all
clean-all: clean ## 深度清理（包括依赖）
	@echo "$(BLUE)深度清理...$(RESET)"
	rm -rf .venv/ uv.lock
	@echo "$(GREEN)深度清理完成!$(RESET)"


# =============================================================================
# 项目信息相关命令
# =============================================================================

.PHONY: info
info: ## 显示项目信息
	@echo "$(CYAN)项目信息$(RESET)"
	@echo "$(CYAN)========$(RESET)"
	@echo "$(YELLOW)项目名称:$(RESET) $(PROJECT_NAME)"
	@echo "$(YELLOW)Python版本:$(RESET) $$($(UV) run $(PYTHON) --version)"
	@echo "$(YELLOW)UV版本:$(RESET) $$($(UV) --version)"
	@echo "$(YELLOW)项目路径:$(RESET) $$(pwd)"
	@echo "$(YELLOW)虚拟环境:$(RESET) $$($(UV) run which $(PYTHON))"
	@echo ""

.PHONY: deps
deps: ## 显示依赖信息
	@echo "$(CYAN)依赖信息$(RESET)"
	@echo "$(CYAN)========$(RESET)"
	@echo "$(YELLOW)主要依赖:$(RESET)"
	$(UV) run $(PYTHON) -c "import importlib.metadata; deps=['duckdb', 'gradio', 'pandas', 'pillow', 'pyarrow']; [print(f'  {dep}=={importlib.metadata.version(dep)}') for dep in deps if importlib.metadata.version(dep)]" 2>/dev/null || echo "  依赖信息获取失败，请运行 'make show-deps' 查看完整依赖树"
	@echo ""

# =============================================================================
# 快速开发命令组合
# =============================================================================

.PHONY: dev-setup
dev-setup: install ## 开发环境快速设置
	@echo "$(GREEN)开发环境设置完成!$(RESET)"
	@echo "$(YELLOW)现在可以运行:$(RESET)"
	@echo "  make test        # 运行测试"
	@echo "  make demo        # 启动demo"
	@echo "  make format      # 格式化代码"

.PHONY: quick-test
quick-test: clean test ## 快速测试（清理 + 测试）
	@echo "$(GREEN)快速测试完成!$(RESET)"


# =============================================================================
# 特殊目标
# =============================================================================

.PHONY: all
all: clean install test build ## 执行完整的构建流程
	@echo "$(GREEN)完整构建流程完成!$(RESET)"

# 防止将文件当作目标
.PHONY: $(shell awk -F: '/^[a-zA-Z_-]+:/ {print $$1}' $(MAKEFILE_LIST))
