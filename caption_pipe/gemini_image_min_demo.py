"""
export GEMINI_API_KEY=ak-dhco9tfkhr9sb5n2zkgoy0alyeodm3ig

python /data/benchmark_metrics/caption_pipe/gemini_image_min_demo.py \
  --content /data/benchmark_metrics/assets/content.webp \
  --style /data/benchmark_metrics/assets/style.webp \
  --prompt "Transfer the style while keeping the content." \
  --out /data/benchmark_metrics/logs/result.png \
  --model_id "gemini-2.5-flash-image-native"
"""
import argparse
import os
from io import BytesIO

from google import genai
from google.genai import types
from PIL import Image


def image_to_part(path: str) -> types.Part:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="gemini-3-pro-native")
    ap.add_argument("--base_url", default="https://models-proxy.stepfun-inc.com/gemini")
    ap.add_argument("--api_version", default="v1alpha")
    args = ap.parse_args()

    client = genai.Client(
        http_options={"api_version": args.api_version, "base_url": args.base_url},
        api_key=os.getenv("GEMINI_API_KEY", ""),
    )

    content_part = image_to_part(args.content)
    style_part = image_to_part(args.style)
    text_part = types.Part.from_text(text=args.prompt)

    resp = client.models.generate_content(
        model=args.model_id,
        contents=[content_part, style_part, text_part],
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    saved = False
    for part in resp.parts:
        if part.inline_data is not None:
            img = part.as_image()
            print(f"save image to {args.out}")
            img.save(args.out)
            saved = True
            break
    if not saved:
        raise SystemExit("No image returned")


if __name__ == "__main__":
    main()
