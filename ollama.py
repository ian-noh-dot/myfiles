import json
import re
import urllib.error
import urllib.request
from pathlib import Path

# --- SETTINGS ---
SPLIT_DIR = Path(r"C:\Users\iansi\Desktop\codetext\split_output")
OUTPUT_JSONL = Path(r"C:\Users\iansi\Desktop\codetext\ollama_dataset.jsonl")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-coder:6.7b"

# Skip empty or ultra-short files (e.g., stray separators/headers)
MIN_CHARS = 40

# Keep snippets concise so local inference stays faster and times out less
MAX_CHARS_PER_SNIPPET = 3000

# Optional batch cap (set to None to run until stopped or finished)
MAX_NEW_FILES = None

# Timeout per request in seconds
TIMEOUT_SECONDS = 90

# Filenames/substrings indicating non-code docs, templates, or directory listings
NON_CODE_PATTERNS = re.compile(
    r"(readme|license|licence|security|contributing|code_of_conduct|changelog|"
    r"dir_list|filelist|file_list|issue_template|pull_request_template|\.md\b)",
    re.IGNORECASE,
)


def is_non_code_or_empty(file_path: Path, text: str) -> tuple[bool, str]:
    if not text or len(text) < MIN_CHARS:
        return True, "empty or too short"
    if NON_CODE_PATTERNS.search(file_path.name):
        return True, "non-code doc/list/template"
    return False, ""


def load_already_processed(jsonl_path: Path) -> set[str]:
    processed = set()
    if not jsonl_path.exists():
        return processed

    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "file" in obj:
                    processed.add(obj["file"])
            except json.JSONDecodeError:
                continue
    return processed


def build_prompt(filename: str, code_text: str) -> str:
    return (
        "Analyze the following source code objectively and describe what this code does "
        "in one concise sentence (mentioning its language, key actions, or logic). "
        "Do not add extra commentary or markdown.\n\n"
        f"File: {filename}\n"
        "Code:\n"
        f"{code_text}"
    )


def to_instruction(what_it_does: str, filename: str) -> str:
    cleaned = " ".join(what_it_does.strip().strip('"').strip("'").split())
    if not cleaned:
        return f"Explain or implement the behavior in {filename}."
    return f"Write code that performs the following behavior: {cleaned}"


def ask_ollama(prompt: str, model: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "").strip()


def main():
    if not SPLIT_DIR.exists():
        print(f"Split folder not found: {SPLIT_DIR}")
        return

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    processed = load_already_processed(OUTPUT_JSONL)
    txt_files = sorted(SPLIT_DIR.glob("*.txt"))

    print(f"Already in JSONL: {len(processed)} | Total split files: {len(txt_files)}\n", flush=True)

    new_count = 0
    skipped_count = 0

    with open(OUTPUT_JSONL, "a", encoding="utf-8", newline="\n") as out_f:
        for file_path in txt_files:
            if file_path.name in processed:
                continue

            if MAX_NEW_FILES is not None and new_count >= MAX_NEW_FILES:
                print(f"\nReached MAX_NEW_FILES ({MAX_NEW_FILES}). Stopping.", flush=True)
                break

            raw_text = file_path.read_text(encoding="utf-8", errors="replace").strip()
            skip, reason = is_non_code_or_empty(file_path, raw_text)
            if skip:
                skipped_count += 1
                continue

            truncated = False
            code_text = raw_text
            if len(code_text) > MAX_CHARS_PER_SNIPPET:
                code_text = code_text[:MAX_CHARS_PER_SNIPPET]
                truncated = True

            prompt = build_prompt(file_path.name, code_text)

            try:
                what_it_does = ask_ollama(prompt, MODEL_NAME)
            except urllib.error.URLError as e:
                print(f"File: {file_path.name}", flush=True)
                print(f"Ollama: [CONNECTION ERROR] {e}", flush=True)
                print("-" * 50, flush=True)
                print("Make sure Ollama is running, then re-run.", flush=True)
                break
            except Exception as e:
                print(f"File: {file_path.name}", flush=True)
                print(f"Ollama: [ERROR] {e}", flush=True)
                print("-" * 50, flush=True)
                continue

            instruction = to_instruction(what_it_does, file_path.name)

            print(f"File: {file_path.name}", flush=True)
            print(f"Ollama: {instruction}", flush=True)
            print("-" * 50, flush=True)

            record = {
                "file": file_path.name,
                "summary": what_it_does,
                "instruction": instruction,
                "output": code_text,
                "truncated": truncated,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            new_count += 1
            processed.add(file_path.name)

    print(
        f"\nDone! Added {new_count} new records (skipped {skipped_count} non-code/empty files) -> {OUTPUT_JSONL}",
        flush=True,
    )


if __name__ == "__main__":
    main()