import pickle
import uuid
from typing import final

import xxhash


@final
class ID:
    __slots__ = ("_value",)

    def __init__(self, value: bytes) -> None:
        if not isinstance(value, bytes) or len(value) != 16:
            raise ValueError(f"value must be exactly 16 bytes, got {value}")
        self._value: bytes = value

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ID):
            return False
        return self._value == other._value

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._value.hex()})"

    def __str__(self) -> str:
        return self._value.hex()

    def to_bytes(self) -> bytes:
        return self._value

    def to_int(self) -> int:
        return int.from_bytes(self._value, byteorder="big", signed=False)

    def to_uuid(self) -> uuid.UUID:
        return uuid.UUID(bytes=self._value)

    @classmethod
    def from_hex(cls, hex_str: str) -> "ID":
        return cls(bytes.fromhex(hex_str))

    @classmethod
    def from_int(cls, i: int):
        return cls(i.to_bytes(16, byteorder="big", signed=False))

    @classmethod
    def from_uuid(cls, u: uuid.UUID | str) -> "ID":
        if isinstance(u, str):
            u = uuid.UUID(u)
        return cls(u.bytes)

    @classmethod
    def from_string(cls, s: str) -> "ID":
        if "-" in s:
            return cls.from_uuid(s)
        if s.startswith("0x"):
            s = s[2:]
        return cls.from_hex(s)

    @classmethod
    def from_(cls, value: str | bytes | uuid.UUID) -> "ID":
        if isinstance(value, ID):
            return value
        elif isinstance(value, str):
            return cls.from_string(value)
        elif isinstance(value, bytes):
            return cls(value)
        elif isinstance(value, uuid.UUID):
            return cls(value.bytes)
        else:
            raise ValueError(f"invalid ID value: {value}")

    @classmethod
    def random(cls) -> "ID":
        return cls(uuid.uuid4().bytes)

    @classmethod
    def hash(cls, *x) -> "ID":
        if len(x) == 1 and isinstance(x[0], bytes):
            return ID(xxhash.xxh3_128_digest(x[0]))

        return ID(xxhash.xxh3_128_digest(pickle.dumps(x, protocol=4)))
