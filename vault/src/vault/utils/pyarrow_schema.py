import base64

import pyarrow as pa
import pyarrow.ipc as ipc


def serialize_pyarrow_schema(schema: pa.Schema) -> dict[str, str]:
    serialized_buffer = schema.serialize()
    base64_string = base64.b64encode(serialized_buffer.to_pybytes()).decode("utf-8")

    human_readable_string = str(schema)

    json_payload = {
        "human_readable": human_readable_string,
        "schema_base64": base64_string,
    }

    return json_payload


def deserialize_pyarrow_schema(payload: dict[str, str]) -> pa.Schema:
    base64_string = payload["schema_base64"]
    buffer = pa.py_buffer(base64.b64decode(base64_string.encode("utf-8")))
    return ipc.read_schema(buffer)
