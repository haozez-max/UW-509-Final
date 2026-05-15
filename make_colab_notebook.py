import json
from pathlib import Path

import nbformat as nbf


SOURCE = Path("ppml_part2_colab_experiments.py")
TARGET = Path("PPML_Part2_Colab_Experiments.ipynb")


def split_percent_script(source_text):
    cells = []
    current_kind = "code"
    current_lines = []

    def flush():
        if not current_lines:
            return
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
            cells.append(nbf.v4.new_markdown_cell("\n".join(markdown_lines).strip()))
        else:
            cells.append(nbf.v4.new_code_cell(text))

    for line in source_text.splitlines(keepends=True):
        if line.startswith("# %%"):
            flush()
            current_lines = []
            current_kind = "markdown" if "[markdown]" in line else "code"
            continue
        current_lines.append(line)
    flush()
    return cells


def main():
    source_text = SOURCE.read_text(encoding="utf-8")
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = split_percent_script(source_text)
    notebook["metadata"] = {
        "colab": {
            "name": TARGET.name,
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    nbf.validate(notebook)
    TARGET.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
