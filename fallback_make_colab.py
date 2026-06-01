import json

source_text = open("ppml_part3_colab_experiments.py", "r", encoding="utf-8").read()

cells = []
current_kind = "code"
current_lines = []

def flush():
    if not current_lines: return
    text = "".join(current_lines).strip("\n")
    if current_kind == "markdown":
        markdown_lines = []
        for line in text.splitlines():
            if line.startswith("# "):
                markdown_lines.append(line[2:])
            elif line.startswith("#"):
                markdown_lines.append(line[1:].lstrip())
            else:
                markdown_lines.append(line)
        src = [line + "\n" for line in "\n".join(markdown_lines).strip().splitlines()]
        # Remove trailing newline from last line to match standard Jupyter
        if src and src[-1].endswith("\n"): src[-1] = src[-1][:-1]
        cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
    else:
        src = [line + "\n" for line in text.splitlines()]
        if src and src[-1].endswith("\n"): src[-1] = src[-1][:-1]
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src})

for line in source_text.splitlines(keepends=True):
    if line.startswith("# %%"):
        flush()
        current_lines = []
        current_kind = "markdown" if "[markdown]" in line else "code"
        continue
    current_lines.append(line)
flush()

notebook = {
  "cells": cells,
  "metadata": {
    "colab": {"name": "PPML_Part3_Colab_Experiments.ipynb", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"}
  },
  "nbformat": 4,
  "nbformat_minor": 0
}

with open("PPML_Part3_Colab_Experiments.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Updated PPML_Part3_Colab_Experiments.ipynb")
