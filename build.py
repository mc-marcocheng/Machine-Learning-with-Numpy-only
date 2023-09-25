import re
import subprocess
from pathlib import Path

tag_pattern = re.compile(r"#\s*tag::(.+?)\n(.+?)\n#\s*end::\1", re.DOTALL)

tag_index = {}
for filename in Path().rglob("*.py"):
    with open(filename, "r", encoding="utf-8") as f:
        code = f.read()
    if m := re.findall(tag_pattern, code):
        for tag, content in m:
            tag_index[tag] = content

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
                output = (
                    output.decode("utf-8").replace("\r\n", "\n").rstrip().rstrip("\n")
                )
                print(output.encode())
                template = template.replace(
                    pattern, detail.format(" open" if collapse == "+" else "", output)
                )
    with open(
        filename.parent / filename.name.replace(".template", "", 1),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(template)
