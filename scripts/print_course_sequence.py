from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "course_manifest.json").read_text(encoding="utf-8"))
    print(data["title"])
    print(data["goal"])
    print()
    print("stage | docs | exercise | lesson answer")
    print("---: | --- | --- | ---")
    for module in data["modules"]:
        print(
            f"{module['stage']:02d} | {module['doc']} | "
            f"{module['exercise']} | {module['lesson']}"
        )


if __name__ == "__main__":
    main()
