import argparse
import base64
import os
import requests
from io import BytesIO
from PIL import Image
ASPECT_RATIO=["2048x2048",
    "2304x1728",
    "1728x2304",
    "2848x1600",
    "1600x2848",
    "2496x1664",
    "1664x2496",
    "3136x1344",
    "3072x3072",
    "3456x2592",
    "2592x3456",
    "4096x2304",
    "2304x4096",
    "2496x3744",
    "3744x2496",
    "4704x2016",
]

def convert_image_to_base64(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def convert_base64_to_image(data: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cref", required=True)
    ap.add_argument("--sref", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="doubao-seedream-4.5")
    ap.add_argument("--base_url", default="https://models-proxy.stepfun-inc.com/v1/images/generations")
    ap.add_argument("--size", default="1600x2848")
    args = ap.parse_args()

    api_key = os.getenv("SEEDREAM_API_KEY", "")
    if not api_key:
        raise SystemExit("SEEDREAM_API_KEY is required")

    cref_image = Image.open(args.cref).convert("RGB")
    sref_image = Image.open(args.sref).convert("RGB")
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "images": [convert_image_to_base64(cref_image), convert_image_to_base64(sref_image)],
        "size": args.size,
        "response_format": "b64_json",
        "sequential_image_generation": "auto",
        "sequential_image_generation_options": {"max_images": 1},
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    resp = requests.post(args.base_url, headers=headers, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    url = data["data"][0]["url"]
    image_resp = requests.get(url, timeout=300)
    image_resp.raise_for_status()
    output_image = Image.open(BytesIO(image_resp.content)).convert("RGB")
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    output_image.save(args.out)


if __name__ == "__main__":
    main()
            
