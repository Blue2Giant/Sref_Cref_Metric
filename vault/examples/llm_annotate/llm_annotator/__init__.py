"""
Annotation Core - Stable utilities for vault annotation tasks.

This package provides:
- Base classes: AnnotateTool, VaultAnnotator
- Common utilities: Image encoding, API retry, JSON validation
- Prompt template system with Jinja2

API Stability Guarantee:
- Core interfaces remain stable
- Safe to build custom tools on top
- See examples in ../tools/ directory

Usage:
    from llm_annotate.core import AnnotateTool, VaultAnnotator
    from llm_annotate.core.utils import call_vlm_single
    from llm_annotate.core.prompt_loader import load_prompt

    class MyTool(AnnotateTool):
        basic_name = "my_tool"
        ...
"""

from .base import AnnotateTool, VaultAnnotator

__all__ = ["AnnotateTool", "VaultAnnotator"]
