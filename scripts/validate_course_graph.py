from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def numbered(path: Path, suffix: str) -> set[str]:
    pat = re.compile(r"^(\d{2})_.+" + re.escape(suffix) + r"$")
    out: set[str] = set()
    for p in path.glob(f"*{suffix}"):
        if pat.match(p.name):
            out.add(str(p.relative_to(ROOT)).replace("\\", "/"))
    return out


def main() -> int:
    manifest = json.loads((ROOT / "course_manifest.json").read_text(encoding="utf-8"))
    modules = manifest.get("modules", [])
    errors: list[str] = []

    docs = {m["doc"] for m in modules}
    exercises = {m["exercise"] for m in modules}
    lessons = {m["lesson"] for m in modules}

    for m in modules:
        for key in ("doc", "exercise", "lesson"):
            p = ROOT / m[key]
            if not p.exists():
                errors.append(f"missing {key}: {m[key]}")

        doc_text = (ROOT / m["doc"]).read_text(encoding="utf-8") if (ROOT / m["doc"]).exists() else ""
        ex_text = (ROOT / m["exercise"]).read_text(encoding="utf-8") if (ROOT / m["exercise"]).exists() else ""
        if "## 新出用語" not in doc_text:
            errors.append(f"doc lacks 新出用語 section: {m['doc']}")
        if m["exercise"] not in doc_text:
            errors.append(f"doc does not mention exercise path: {m['doc']} -> {m['exercise']}")
        if m["lesson"] not in doc_text:
            errors.append(f"doc does not mention lesson path: {m['doc']} -> {m['lesson']}")
        if m["doc"] not in ex_text:
            errors.append(f"exercise does not mention doc path: {m['exercise']} -> {m['doc']}")
        if m["lesson"] not in ex_text:
            errors.append(f"exercise does not mention lesson path: {m['exercise']} -> {m['lesson']}")
        if "解答" not in ex_text:
            errors.append(f"exercise does not mark lesson as answer: {m['exercise']}")

    actual_docs = numbered(ROOT / "docs" / "modules", ".md")
    actual_exercises = numbered(ROOT / "exercises", ".md")
    actual_lessons = numbered(ROOT / "lessons", ".py")

    for extra in sorted(actual_docs - docs):
        errors.append(f"unmapped module doc: {extra}")
    for extra in sorted(actual_exercises - exercises):
        errors.append(f"unmapped exercise: {extra}")
    for extra in sorted(actual_lessons - lessons):
        errors.append(f"unmapped lesson: {extra}")
    for missing in sorted(docs - actual_docs):
        errors.append(f"manifest doc not in docs/modules: {missing}")
    for missing in sorted(exercises - actual_exercises):
        errors.append(f"manifest exercise not in exercises: {missing}")
    for missing in sorted(lessons - actual_lessons):
        errors.append(f"manifest lesson not in lessons: {missing}")

    if errors:
        print("Course graph validation failed:")
        for e in errors:
            print(f"- {e}")
        return 1
    print(f"OK: {len(modules)} modules; every docs/modules file, exercise, and lesson is mapped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
