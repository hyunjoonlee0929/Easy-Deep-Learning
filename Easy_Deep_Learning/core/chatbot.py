"""Lightweight chatbot utilities for summarizing GitHub README files."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

import requests


@dataclass
class ChatbotResult:
    source: str
    title: str
    summary: str
    features: list[str]
    usage: list[str]
    setup: list[str]
    commands: list[str]
    notes: list[str]
    raw_sections: dict[str, list[str]]


def _normalize_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_sections(lines: Iterable[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = "intro"
    sections[current] = []

    for line in lines:
        if line.strip().startswith("#"):
            heading = _normalize_heading(line.strip("# ").strip())
            current = heading or "untitled"
            sections.setdefault(current, [])
        else:
            sections[current].append(line.rstrip())

    return sections


def _collect_bullets(lines: Iterable[str]) -> list[str]:
    bullets: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("-", "*", "+")):
            bullets.append(stripped.lstrip("-*+ ").strip())
    return [b for b in bullets if b]


def _collect_code_blocks(text: str, max_blocks: int = 6) -> list[str]:
    blocks: list[str] = []
    fence_pattern = re.compile(r"```(.*?)```", re.DOTALL)
    for match in fence_pattern.finditer(text):
        block = match.group(1).strip()
        if block:
            blocks.append(block)
        if len(blocks) >= max_blocks:
            break
    return blocks


def _heuristic_summary(readme_text: str, source: str) -> ChatbotResult:
    lines = readme_text.splitlines()
    sections = _extract_sections(lines)

    title = ""
    for line in lines:
        if line.strip().startswith("# "):
            title = line.strip("# ").strip()
            break
    if not title:
        title = "README Summary"

    intro_lines = sections.get("intro", [])
    summary = ""
    for line in intro_lines:
        if line.strip():
            summary = line.strip()
            break
    if not summary:
        summary = "No explicit overview found. Summarized key sections below."

    def pick_section(keys: list[str]) -> list[str]:
        collected: list[str] = []
        for key, content in sections.items():
            normalized = _normalize_heading(key)
            if any(k in normalized for k in keys):
                collected.extend(_collect_bullets(content))
        return [c for c in collected if c]

    features = pick_section(["feature", "기능", "overview", "핵심"])
    usage = pick_section(["usage", "quick start", "사용법", "실행", "run"])
    setup = pick_section(["install", "setup", "설치", "requirements"])

    all_bullets = _collect_bullets(lines)
    if not features:
        features = all_bullets[:6]
    if not usage:
        usage = all_bullets[6:12]
    if not setup:
        setup = all_bullets[12:18]

    commands = _collect_code_blocks(readme_text)
    notes: list[str] = []
    if not commands:
        notes.append("No code blocks detected in README.")

    return ChatbotResult(
        source=source,
        title=title,
        summary=summary,
        features=features,
        usage=usage,
        setup=setup,
        commands=commands,
        notes=notes,
        raw_sections=sections,
    )


def _openai_summary(readme_text: str, source: str) -> ChatbotResult | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None

    client = OpenAI()
    prompt = (
        "Summarize the README. Return STRICT JSON with keys: "
        "title (string), summary (string), features (array of strings), "
        "usage (array), setup (array), commands (array), notes (array)."
    )
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": "You summarize GitHub READMEs for users."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": readme_text[:12000]},
        ],
    )
    text = ""
    for item in response.output:
        if item.type == "output_text":
            text += item.text
    if not text:
        return None
    try:
        payload = json.loads(text)
    except Exception:
        return None

    return ChatbotResult(
        source=source,
        title=str(payload.get("title", "README Summary")),
        summary=str(payload.get("summary", "")),
        features=[str(x) for x in payload.get("features", [])],
        usage=[str(x) for x in payload.get("usage", [])],
        setup=[str(x) for x in payload.get("setup", [])],
        commands=[str(x) for x in payload.get("commands", [])],
        notes=[str(x) for x in payload.get("notes", [])],
        raw_sections={},
    )


def _github_raw_url(owner: str, repo: str, branch: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


def _parse_github_url(url: str) -> tuple[str, str, str | None]:
    match = re.match(r"https?://github\.com/([^/]+)/([^/]+)(/.*)?", url)
    if not match:
        raise ValueError("Invalid GitHub URL.")
    owner, repo, tail = match.group(1), match.group(2), match.group(3)
    repo = repo.replace(".git", "")
    if tail and "/blob/" in tail:
        parts = tail.split("/blob/")
        if len(parts) == 2:
            branch_and_path = parts[1].lstrip("/")
            if "/" in branch_and_path:
                branch, path = branch_and_path.split("/", 1)
                return owner, repo, f"{branch}/{path}"
    return owner, repo, None


def fetch_readme_from_github(url: str, timeout: int = 12) -> tuple[str, str]:
    owner, repo, explicit = _parse_github_url(url)
    session = requests.Session()
    session.headers.update({"User-Agent": "EasyDeepLearning-Chatbot"})

    if explicit:
        branch, path = explicit.split("/", 1)
        raw_url = _github_raw_url(owner, repo, branch, path)
        resp = session.get(raw_url, timeout=timeout)
        resp.raise_for_status()
        return resp.text, raw_url

    candidates = [
        ("main", "README.md"),
        ("main", "readme.md"),
        ("main", "README.rst"),
        ("master", "README.md"),
        ("master", "readme.md"),
        ("master", "README.rst"),
    ]
    for branch, path in candidates:
        raw_url = _github_raw_url(owner, repo, branch, path)
        resp = session.get(raw_url, timeout=timeout)
        if resp.status_code == 200 and resp.text.strip():
            return resp.text, raw_url

    raise ValueError("README not found in main/master branches.")


def summarize_readme_text(readme_text: str, source: str = "manual") -> ChatbotResult:
    result = _openai_summary(readme_text, source)
    if result:
        return result
    return _heuristic_summary(readme_text, source)


def summarize_github_readme(url: str) -> ChatbotResult:
    readme_text, source = fetch_readme_from_github(url)
    return summarize_readme_text(readme_text, source=source)


def fetch_repo_contents(url: str, timeout: int = 12) -> list[dict[str, Any]]:
    owner, repo, _ = _parse_github_url(url)
    api = f"https://api.github.com/repos/{owner}/{repo}/contents"
    session = requests.Session()
    session.headers.update({"User-Agent": "EasyDeepLearning-Chatbot"})
    resp = session.get(api, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    return payload if isinstance(payload, list) else []


def analyze_repo_structure(url: str) -> dict[str, Any]:
    contents = fetch_repo_contents(url)
    names = [item.get("name", "") for item in contents]
    paths = [item.get("path", "") for item in contents]

    key_files = []
    for name in ["README.md", "requirements.txt", "pyproject.toml", "package.json", "main.py", "app.py"]:
        if name in names:
            key_files.append(name)

    tech = []
    if "requirements.txt" in names or "pyproject.toml" in names:
        tech.append("python")
    if "package.json" in names:
        tech.append("node")

    commands = []
    if "requirements.txt" in names:
        commands.append("pip install -r requirements.txt")
    if "main.py" in names:
        commands.append("python main.py --help")
    if "app.py" in names or any("dashboard" in p for p in paths):
        commands.append("streamlit run app.py")
    if any("api" in p for p in paths) or "app.py" in paths:
        commands.append("uvicorn api.app:app --reload --port 8000")

    return {
        "key_files": key_files,
        "tech_stack": tech,
        "commands": commands,
        "file_count": len(paths),
        "files": paths[:50],
    }


def summarize_github_repo(url: str) -> dict[str, Any]:
    summary = summarize_github_readme(url)
    try:
        repo_info = analyze_repo_structure(url)
    except Exception as exc:
        repo_info = {"error": str(exc)}
    return {
        "title": summary.title,
        "summary": summary.summary,
        "features": summary.features,
        "usage": summary.usage,
        "setup": summary.setup,
        "commands": summary.commands,
        "repo_info": repo_info,
        "source": summary.source,
    }

def extract_github_urls(text: str) -> list[str]:
    pattern = re.compile(r"https?://github\.com/[^\s)]+")
    return pattern.findall(text)


def _fallback_chat(message: str) -> str:
    msg = message.strip().lower()
    if "help" in msg or "사용법" in msg:
        return (
            "GitHub 링크를 입력하면 README를 요약합니다. "
            "예: https://github.com/owner/repo"
        )
    if "summary" in msg or "요약" in msg:
        urls = extract_github_urls(message)
        if urls:
            try:
                result = summarize_github_readme(urls[0])
                return f"[{result.title}]\n{result.summary}"
            except Exception as exc:
                return f"요약 실패: {exc}"
        return "요약할 GitHub 링크를 함께 보내주세요."
    return "질문을 구체적으로 알려주세요. GitHub 링크 요약을 지원합니다."


def chat_response(message: str) -> str:
    """Return a chatbot response using OpenAI if available, else a deterministic fallback."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_chat(message)
    try:
        from openai import OpenAI
    except Exception:
        return _fallback_chat(message)

    client = OpenAI()
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": "You are a helpful assistant for Easy Deep Learning users."},
            {"role": "user", "content": message},
        ],
    )
    text = ""
    for item in response.output:
        if item.type == "output_text":
            text += item.text
    if text.strip():
        return text.strip()

    urls = extract_github_urls(message)
    if urls:
        try:
            info = summarize_github_repo(urls[0])
            return (
                f"{info['title']}\n{info['summary']}\n"
                f"Features: {', '.join(info.get('features', [])[:5])}\n"
                f"Commands: {', '.join(info.get('repo_info', {}).get('commands', []))}"
            )
        except Exception as exc:
            return f"요약 실패: {exc}"
    return _fallback_chat(message)
