import re
import subprocess
from pathlib import Path

# fix ugly black-modified tag comments
tag_pattern = re.compile(r"(#\s*tag::(.+?)\n(.+?)\n#\s*end::\2)", re.DOTALL)

files_with_tags = []
for filename in Path().rglob("*.py"):
    with open(filename, "r", encoding="utf-8") as f:
        code = f.read()
    if m := re.findall(tag_pattern, code):
        files_with_tags.append(filename)
        for full, tag, content in m:
            new_full = f"# tag::{tag}\n{content.rstrip()}\n# end::{tag}"
            code = code.replace(full, new_full)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)


# get all tags and build index
tag_pattern = re.compile(r"#\s*tag::(.+?)\n(.+?)\n#\s*end::\1", re.DOTALL)

tag_index = {}
for filename in files_with_tags:
    with open(filename, "r", encoding="utf-8") as f:
        code = f.read()
    if m := re.findall(tag_pattern, code):
        for tag, content in m:
            tag_index[tag] = content.rstrip()


# put content in markdown
# pattern ${{ ... }} gets replaced by code
# pattern $[[ ... ]] gets replaced by code's text output as collapsed block
# pattern $[[ +... ]] gets replaced by code's text output as expanded collapsible block
# pattern $[[ -... ]] will run the code only, no outputs will be shown in markdown
template_pattern = re.compile(r"(\${{\s*(.+?)\s*}})")
run_pattern = re.compile(r"(\$\[\[\s*([+-]?)\s*(.+?)\s*\]\])")
detail = """<details{}>
<summary>Output</summary>

```
{}
```

</details>
"""

for filename in Path().rglob("*.template.*"):
    with open(filename, "r", encoding="utf-8") as f:
        template = f.read()
    if m := re.findall(template_pattern, template):
        for pattern, tag in m:
            template = template.replace(pattern, tag_index[tag])
    if m := re.findall(run_pattern, template):
        for pattern, collapse, py_file in m:
            if collapse == "-":
                subprocess.run(["python", "-m", py_file], shell=True)
                template = template.replace(pattern, "")
            else:
                output = subprocess.check_output(["python", "-m", py_file], shell=True)
                output = output.decode("utf-8").replace("\r\n", "\n").rstrip()
                template = template.replace(
                    pattern, detail.format(" open" if collapse == "+" else "", output)
                )
    with open(
        filename.parent / filename.name.replace(".template", "", 1),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(template)
