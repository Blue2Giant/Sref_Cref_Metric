from jinja2 import Template
from openai import OpenAI

from .conversation import build_messages
from .openai_client import call_openai, retry_on_error


def load_prompt_template(template_path: str) -> Template:
    with open(template_path, encoding="utf-8") as f:
        template = Template(f.read())
    return template


def annotate_single_turn(
    images: bytes | list[bytes],
    model: str,
    client: OpenAI,
    prompt: str,
    max_tokens: int = 2048,
    max_pixels: int = 512 * 32 * 32,
    is_json: bool = True,
    extra_body: dict | None = None,
    retry_count: int = 3,
) -> dict | str:
    if isinstance(images, bytes):
        images = [images]
    messages = build_messages(text=prompt, images=images, max_pixels=max_pixels)

    if extra_body is None and len(images) > 0:
        extra_body = {"chat_template_kwargs": {"add_vision_id": True}}

    result = retry_on_error(
        call_openai,
        messages=messages,
        model=model,
        client=client,
        max_tokens=max_tokens,
        extra_body=extra_body,
        is_json=is_json,
        retry_count=retry_count,
    )

    return result


# def annotate_multi_turn(
#     images: bytes | list[bytes],
#     model: str,
#     client: OpenAI,
#     prompt: str,
#     max_turns: int = 2,
#     max_tokens: int = 2048,
#     extra_body: dict | None = None,
#     max_pixels: int = 512 * 32 * 32,
#     is_json: bool = True,
# ) -> list[dict | str]:
#     """
#     multi-turn annotation for single image.
#     """
#     if isinstance(images, bytes):
#         images = [images]
#     if extra_body is None and len(images) > 0:
#         extra_body = {"chat_template_kwargs": {"add_vision_id": True}}

#     conv = Conversation()
#     conv.add_system("You are a helpful assistant.")
#     conv.add_user(text=prompt, images=images, max_pixels=max_pixels)

#     results: list[dict | str] = []
#     for turn in range(max_turns):
#         # 调用 API
#         response = retry_on_error(
#             call_openai,
#             messages=conv.get_messages(),
#             model=model,
#             client=client,
#             max_tokens=max_tokens,
#             extra_body=extra_body,
#             is_json=is_json,
#         )
#         results.append(response)

#         # 检查是否需要追问
#         if not should_followup(response) or turn == max_turns - 1:
#             break

#         # 生成追问
#         followup = load_prompt(
#             "followup.j2", previous_result=json.dumps(response, indent=2)
#         )

#         conv.add_assistant(json.dumps(response))
#         conv.add_user(text=followup)

#     return results
