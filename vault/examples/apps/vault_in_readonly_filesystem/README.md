# DuckDB 只读文件系统 Hack

## 🚀 快速使用（推荐）

我们已经预编译好了 `fcntl_hack.so` 共享库，你可以直接使用：

### 1. 下载预编译文件

```bash
# 使用 megfile 从 S3 复制（推荐）
megfile copy s3://ruiwang/tmp/fcntl_hack.so ./

# 或者使用 wget/curl 下载（如果配置了访问权限）
wget https://your-s3-endpoint/ruiwang/tmp/fcntl_hack.so
```

### 2. 直接使用

```bash
# 设置权限
chmod +x fcntl_hack.so

# 在只读文件系统上读取 DuckDB
LD_PRELOAD=./fcntl_hack.so duckdb -readonly /mnt/marmot/i-liushiyu/hq_v2/metadata.duckdb
```

### 3. Python 脚本中使用

```bash
LD_PRELOAD=./fcntl_hack.so python your_script.py
```

> **注意**：这个预编译版本适用于主流的 Linux x86_64 系统。如果遇到兼容性问题，请参考下面的编译说明自行编译。

---

## 问题描述

当在只读文件系统（如 JFS 只读挂载点、网络存储等）上尝试以只读模式打开 DuckDB 数据库时，会遇到以下错误：

```
Cannot open file "/path/to/metadata.duckdb": read-only filesystem
```

这个错误的根本原因是：即使以只读模式打开数据库，DuckDB 仍然会尝试获取文件锁（`flock` 或 `fcntl`）来确保数据的并发安全性。在只读文件系统上，文件锁操作会失败并返回 `EROFS` (Read-only filesystem) 错误。

## 解决方案

本方案通过 `LD_PRELOAD` 机制拦截 DuckDB 的文件锁调用，将 `EROFS` 错误转换为 `ENOTSUP` (Operation not supported) 错误。DuckDB 接收到 `ENOTSUP` 错误后，会认为文件锁不被支持，从而优雅地降级为无锁模式，继续正常工作。

## 工作原理

### 技术细节

1. **LD_PRELOAD 机制**：通过环境变量 `LD_PRELOAD` 预加载共享库，拦截系统调用
2. **函数劫持**：劫持 `flock()` 和 `fcntl()` 系统调用
3. **错误转换**：将文件锁失败的 `EROFS` (30) 错误转换为 `ENOTSUP` (95) 错误
4. **透明处理**：对应用程序完全透明，无需修改 DuckDB 或应用代码

### 核心逻辑

```c
// 检查是否是文件锁操作失败在只读文件系统上
if (result == -1 && original_errno == EROFS) {
    if ((operation & LOCK_SH) || (operation & LOCK_EX)) {
        errno = ENOTSUP;  // 将 EROFS 改为 ENOTSUP
    }
}
```

## 编译和使用

### 1. 编译共享库

```bash
gcc -shared -fPIC -o fcntl_hack.so fcntl_hack.c -ldl
```

**编译参数说明：**
- `-shared`：生成共享库
- `-fPIC`：生成位置无关代码
- `-ldl`：链接动态链接库

### 2. 使用方法

#### Python 脚本中使用

```bash
LD_PRELOAD=./fcntl_hack.so python your_python_script.py
```

#### 直接使用 DuckDB CLI

```bash
LD_PRELOAD=./fcntl_hack.so duckdb -readonly /path/to/metadata.duckdb
```

#### 在 JFS 环境中使用

```bash
# 在挂载的 JFS 目录上使用
LD_PRELOAD=./fcntl_hack.so python your_script.py

# 或者设置环境变量
export LD_PRELOAD=/path/to/fcntl_hack.so
python your_script.py
```

## 验证效果

### 测试脚本

创建一个简单的 Python 测试脚本验证效果：

```python
import duckdb
import os

# 测试连接（在有 LD_PRELOAD 和没有的情况下分别测试）
db_path = '/mnt/jfs_readonly/metadata.duckdb'  # JFS 挂载路径下的数据库文件

try:
    conn = duckdb.connect(db_path, read_only=True)
    result = conn.execute('SELECT COUNT(*) FROM information_schema.tables').fetchone()
    print(f"连接成功，表数量: {result[0]}")
    conn.close()
except Exception as e:
    print(f"连接失败: {e}")
```

### 预期结果

- **不使用 hack**：`Cannot open file: read-only filesystem`
- **使用 hack**：正常连接，可以执行查询

## 安全性考虑

1. **只读安全**：此 hack 仅拦截文件锁操作，不影响其他文件操作
2. **数据完整性**：只在不支持锁的环境下工作，不会损坏数据
3. **适用范围**：仅适用于只读操作，写入操作仍会正常失败

## 适用场景

- ✅ JFS (JuiceFS) 只读挂载点
- ✅ 网络文件系统（NFS、S3FS）只读访问
- ✅ 分布式文件系统的只读客户端
- ✅ 云存储网关的只读挂载
- ✅ 只读文件系统上的数据分析任务

## 局限性

1. **Linux 专用**：仅适用于 Linux 系统
2. **仅限只读**：只解决只读访问问题，不支持写入
3. **动态链接**：需要应用使用动态链接的 glibc

## 故障排除

### 检查共享库是否正确加载

```bash
LD_PRELOAD=./fcntl_hack.so ldd $(which python)
```

### 调试模式

可以在代码中添加调试输出：

```c
fprintf(stderr, "flock: fd=%d, op=%d, errno=%d\n", fd, operation, original_errno);
```

### 常见错误

1. **找不到共享库**：确保路径正确且有执行权限
2. **编译失败**：检查是否安装了 gcc 和开发工具
3. **仍然报错**：确认是 EROFS 错误，不是其他权限问题

## 替代方案

1. **复制到可写目录**：将数据库复制到可写位置（需要额外空间）
2. **使用 DuckDB 内存模式**：但无法访问持久化数据
3. **修改文件系统挂载**：某些情况下可以调整挂载选项

## 技术参考

- `LD_PRELOAD` 机制：Linux 动态链接器预加载机制
- `errno` 错误码：`EROFS` (30) vs `ENOTSUP` (95)
- DuckDB 源码：文件锁实现和错误处理逻辑
- JFS (JuiceFS)：分布式文件系统的只读客户端特性
- FUSE 文件系统：网络文件系统在文件锁支持方面的限制

---

**注意**：此工具是针对特定场景的技术解决方案，请在充分理解原理的情况下使用。