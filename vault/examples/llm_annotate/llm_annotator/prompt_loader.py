"""
Prompt template loading and rendering with Jinja2.

Supports:
- Loading prompt templates from files
- Jinja2 template rendering with variables
- Template inheritance and includes
"""

from pathlib import Path
from typing import Dict, Any

from jinja2 import Environment, FileSystemLoader, Template


class PromptLoader:
    """
    Prompt template loader with Jinja2 support.

    Usage:
        loader = PromptLoader()
        prompt = loader.load("stepflow_tag.j2")
        rendered = loader.render("stepflow_tag.j2", context={"key": "value"})
    """

    def __init__(self, template_dir: str | Path | None = None):
        """
        Initialize prompt loader.

        Args:
            template_dir: Directory containing prompt templates.
                         If None, you must provide template_dir when calling
                         load_prompt() or render_prompt()
        """
        if template_dir is not None:
            template_dir = Path(template_dir)

        self.template_dir = template_dir

        if template_dir is not None:
            self.env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            self.env = None

    def load(self, template_name: str) -> str:
        """
        Load prompt template without rendering.

        Args:
            template_name: Template filename (e.g., "stepflow_tag.j2")

        Returns:
            Raw template string
        """
        if self.template_dir is None:
            raise ValueError(
                "template_dir not set. Initialize PromptLoader with template_dir or "
                "use load_prompt(template_name, template_dir=...)"
            )

        template_path = self.template_dir / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        return template_path.read_text(encoding="utf-8")

    def get_template(self, template_name: str) -> Template:
        """
        Get Jinja2 template object.

        Args:
            template_name: Template filename

        Returns:
            Jinja2 Template object
        """
        if self.env is None:
            raise ValueError(
                "template_dir not set. Initialize PromptLoader with template_dir"
            )
        return self.env.get_template(template_name)

    def render(self, template_name: str, context: Dict[str, Any] | None = None) -> str:
        """
        Render template with context variables.

        Args:
            template_name: Template filename
            context: Dictionary of variables for template

        Returns:
            Rendered prompt string
        """
        if context is None:
            context = {}

        template = self.get_template(template_name)
        return template.render(**context)


# Global loader instance
_default_loader: PromptLoader | None = None


def get_loader(template_dir: str | Path | None = None) -> PromptLoader:
    """Get or create default prompt loader."""
    global _default_loader
    if _default_loader is None or template_dir is not None:
        _default_loader = PromptLoader(template_dir)
    return _default_loader


def load_prompt(template_name: str, template_dir: str | Path | None = None) -> str:
    """
    Load prompt template (convenience function).

    Args:
        template_name: Template filename
        template_dir: Optional custom template directory

    Returns:
        Raw template string
    """
    loader = get_loader(template_dir)
    return loader.load(template_name)


def render_prompt(
    template_name: str,
    context: Dict[str, Any] | None = None,
    template_dir: str | Path | None = None,
) -> str:
    """
    Render prompt template with context (convenience function).

    Args:
        template_name: Template filename
        context: Template variables
        template_dir: Optional custom template directory

    Returns:
        Rendered prompt string
    """
    loader = get_loader(template_dir)
    return loader.render(template_name, context)
