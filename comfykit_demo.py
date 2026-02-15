import asyncio
from comfykit import ComfyKit

async def main():
    kit = ComfyKit(comfyui_url="http://10.191.20.14:8188")
    result = await kit.execute("/data/LoraPipeline/assets/SDXL_illustrious_magic.json")
    print(result.images)  # ['http://10.201.16.49:8188/view?filename=...']

if __name__ == "__main__":
    asyncio.run(main())