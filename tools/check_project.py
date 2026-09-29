"""Check project structure without importing torch or writing bytecode caches."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def check_sources():
    paths = sorted(ROOT.rglob("*.py"))
    count = 0
    for path in paths:
        if any(part in {".venv", "venv", "__pycache__"} for part in path.parts):
            continue
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        tree = ast.parse(source)
        # The preprocessing command has its own utils package beside its entry point.
        base = ROOT / "dataset" if ROOT / "dataset" in path.parents else SRC
        local_names = {p.stem for p in base.glob("*.py")}
        local_names.update(p.name for p in base.iterdir() if p.is_dir())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                if node.module.split(".")[0] in local_names:
                    target = base.joinpath(*node.module.split("."))
                    assert (
                        target.with_suffix(".py").is_file() or (target / "__init__.py").is_file()
                    ), (path, node.module)
        count += 1
    return count


def check_compatibility():
    tree = ast.parse((SRC / "text.py").read_text(encoding="utf-8"))
    exported = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets)
    )
    imported = {
        alias.name for node in tree.body if isinstance(node, ast.ImportFrom) for alias in node.names
    }
    assert set(exported) <= imported
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            target = SRC.joinpath(*node.module.split(".")).with_suffix(".py")
            module = ast.parse(target.read_text(encoding="utf-8"))
            definitions = {
                item.name
                for item in module.body
                if isinstance(item, (ast.FunctionDef, ast.ClassDef))
            }
            assert all(alias.name in definitions for alias in node.names), node.module
    assert {"Inference", "main", "feature_gene", "change_path"} <= set(exported)


def check_paths():
    # Execute the dependency-free helper directly without importing the ML stack.
    path = SRC / "utils" / "data_paths.py"
    scope = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), scope)
    change_path = scope["change_path"]
    cases = [
        ("RSITMD", "data/imgs/001.jpg", "data/imgs/001.jpg"),
        ("ICFG-PEDES", "data/train/001.jpg", "data/train_occlusion_new/001.jpg"),
        ("ICFG-PEDES", "data/test/001.jpg", "data/test_occlusion_new/001.jpg"),
        ("CUHK-PEDES", "data/cam_a/001.jpg", "data/cam_a_occlusion_new/001.jpg"),
        ("CUHK-PEDES", "data/other/001.jpg", "data/other/001.jpg"),
    ]
    for dataset, source, expected in cases:
        assert change_path(dataset, source) == expected


if __name__ == "__main__":
    count = check_sources()
    check_compatibility()
    check_paths()
    print(
        "PASS: {} Python files; local imports, compatibility exports and dataset paths.".format(
            count
        )
    )
