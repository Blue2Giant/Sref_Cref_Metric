"""
Annotation Tools - User-extensible annotation implementations.

This directory contains concrete annotation tool implementations.
Users can copy and modify these as templates for new tasks.

Available tools:
- stepflow_tag_tool: Single image tagging for movie scenes
- (Add your own tools here)

Usage:
    from tools.stepflow_tag_tool import StepflowTagTool
    tool = StepflowTagTool(model_name="qwen3-vl-30ba3b")
"""
