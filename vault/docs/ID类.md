# ID Class

`ID` 类是一个用于表示一个唯一的、16字节（128位）标识符的最终类(final class)。它提供了一种统一的方式来处理各种来源的标识符，例如 UUID、哈希值或数据库记录ID。

这个类的设计目标是：
*   **类型安全**: 确保所有ID都是16字节的 bytes，避免混用不同格式的字符串或整数。
*   **不可变性**: ID对象一旦创建就不能被修改。
*   **易于转换**: 可以轻松地在 `bytes`, `hex`, `int` 和 `uuid.UUID` 对象之间进行转换。
*   **灵活创建**: 支持从多种数据类型和格式创建ID。

## 核心创建方法

`ID` 类主要通过两种方式创建：**随机生成**和**内容哈希**。

*   `ID.random()`: 创建一个随机的ID，内部使用 `uuid.uuid4()`，适用于需要唯一标识符的场景。
*   `ID.hash(*x)`: 根据输入内容创建一个确定性的ID。只要输入相同，输出的ID就永远相同。这在需要为文件、数据块或任何可序列化对象生成一个稳定且唯一的标识符时非常有用。它内部使用高效的 `xxhash.xxh3_128_digest` 算法。

## 使用代码示例

下面是一些典型的使用场景和代码示例。


```python
import uuid

# 引入ID类
from vault.schema import ID

# ---------------------------------------------------------------------------
# 场景1: 创建随机ID (类似于 UUID)
# 当你需要为新实体（如用户、文章、日志条目）生成一个唯一标识符时，使用 random() 方法。
# 这等同于创建一个UUIDv4。
random_id = ID.random()

print(f"随机生成的ID: {random_id}")
print(f"转换回UUID对象: {random_id.to_uuid()}")
print("-" * 20)


# ---------------------------------------------------------------------------
# 场景2: 根据内容创建确定性ID (Hash)
# 当你需要根据某些内容（如文件名、用户邮箱、一段数据）生成一个稳定且唯一的ID时，使用 hash() 方法。
# 只要输入内容不变，输出的ID就永远一样。

# 为字符串生成ID
user_email = "test@example.com"
# ID.hash() 会自动处理对象的序列化
user_id = ID.hash(user_email)

print(f"从字符串 '{user_email}' 生成的哈希ID: {user_id}")

# 再次使用相同内容生成ID，结果是完全一样的
user_id_again = ID.hash("test@example.com")
print(f"确认两次哈希结果相同: {user_id == user_id_again}")


# 为字节数据生成ID
file_content = b"some binary data content"
file_id = ID.hash(file_content)
print(f"从字节数据生成的哈希ID: {file_id}")
print("-" * 20)

```


`ID` 类提供了强大的 `from_` 系列方法，可以方便地从不同格式的现有数据创建 `ID` 对象。

统一入口是 `ID.from_()` 方法，它可以智能地识别输入类型。

```python
# ---------------------------------------------------------------------------
# 场景3: 从不同格式的字符串创建ID

# a) 从标准的UUID格式字符串
uuid_str = "123e4567-e89b-12d3-a456-426614174000"
id_from_uuid_str = ID.from_(uuid_str)
print(f"从UUID字符串创建: {id_from_uuid_str}")

# b) 从16进制字符串 (32个字符)
hex_str = "123e4567e89b12d3a456426614174000"
id_from_hex_str = ID.from_(hex_str)
print(f"从Hex字符串创建: {id_from_hex_str}")

# 确认两者是同一个ID
assert id_from_uuid_str == id_from_hex_str

# ---------------------------------------------------------------------------
# 场景4: 从其他数据类型创建ID

# a) 从 bytes (必须是16字节)
# 直接从正确的hex_str转换，避免手写bytes字面量出错
byte_val = bytes.fromhex(hex_str)
id_from_bytes = ID.from_(byte_val)
print(f"从bytes创建: {id_from_bytes}")

# b) 从 uuid.UUID 对象
uuid_obj = uuid.UUID(uuid_str)
id_from_uuid_obj = ID.from_(uuid_obj)
print(f"从UUID对象创建: {id_from_uuid_obj}")

# c) 从整数 (不常用，但支持)
int_val = int.from_bytes(byte_val, byteorder="big")
id_from_int = ID.from_int(int_val)
print(f"从整数创建: {id_from_int}")


# 确认所有方式创建的ID都相同
assert id_from_uuid_str == id_from_bytes == id_from_uuid_obj == id_from_int
print("\n所有从不同源创建的ID都验证一致！")
print("-" * 20)

```

### 4. ID对象的转换与使用

创建 `ID` 对象后，可以轻松地将其转换回其他格式。

```python
# 我们使用上面创建的 id_from_hex_str 对象
my_id = ID.from_hex("123e4567e89b12d3a456426614174000")

# 获取16进制字符串表示 (默认的字符串转换)
hex_representation = str(my_id)
print(f"Hex表示: {hex_representation}")

# 获取16字节的 bytes 表示
bytes_representation = my_id.to_bytes()
print(f"Bytes表示: {bytes_representation}")

# 获取 uuid.UUID 对象
uuid_representation = my_id.to_uuid()
print(f"UUID对象表示: {uuid_representation}")

# 获取整数表示
int_representation = my_id.to_int()
print(f"整数表示: {int_representation}")

# ID对象可以作为字典的键 (因为它实现了 __hash__ 和 __eq__)
id_map = {my_id: "这是一个示例值"}
print(f"\nID作为字典键: {id_map[my_id]}")

```