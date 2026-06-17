---
name: python-docs-expert
description: Specialized in documenting Python scripts following standard protocols like NumPy. Use when the user needs to add or update docstrings for functions, classes, or modules, especially for scientific and data processing projects like WeatherSoilDataProcessor.
---

# Python Documentation Expert Skill

This skill provides guidance and tools for documenting Python code using the **NumPy Style**, which is the standard for scientific and data projects.

## Workflow

1.  **Analyze the Code**: Read the function or class signature and its implementation to understand its inputs, outputs, and side effects.
2.  **Reference Standards**: Consult [references/numpy-style-guide.md](references/numpy-style-guide.md) for the structure and [references/project-conventions.md](references/project-conventions.md) for project-specific data types (e.g., `OmegaConf`, `pd.DataFrame`, spatial extents).
3.  **Draft the Docstring**: Write a clear, concise docstring. Ensure all parameters and return values are accurately typed and described.
4.  **Apply to File**: Use the `replace` tool to insert the docstring immediately after the definition line (e.g., `def my_func():`).

## Docstring Structure (NumPy Style)

A well-formatted docstring should include:
- A one-line summary.
- A `Parameters` section describing each argument with its type.
- A `Returns` section describing the return value(s) and their type(s).
- (Optional) `Raises` section if exceptions are explicitly raised.
- (Optional) `Examples` section with standard Python REPL formatting (`>>>`).

## Best Practices for this Project

- **Config Objects**: Document `OmegaConf` parameters by specifying which nested keys are expected (e.g., `config.SPATIAL_INFO.extent`).
- **Spatial Data**: Always clarify the format and units for bounding boxes or coordinates.
- **DataFrames**: Mention required columns if the function expects a specific schema.
- **Internal Paths**: Document any temporary directories or files created during execution.

## Tooling

You can use `scripts/generate_docstring_template.py` as a reference for how to structure a programmatic template for any given function signature.
