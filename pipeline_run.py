"""Run the full data pipeline in order, one step at a time.

Order:
    1. processor_data.py      --action all   (scrape raw market data into data/)
    2. upload_csv.py                         (push industries_data + companies_data CSVs to MongoDB, de-duplicated)
    3. processor_documents.py                (fetch IR documents -> JSON)
    4. processor_news.py                     (collect coverage news -> JSON)
    5. processor_press.py                    (collect press releases -> JSON)
    6. upload_json.py                        (push the 3 fetch_*_companies.json to MongoDB)

Usage (from the project root):
    python pipeline_run.py                 # run every step in order
    python pipeline_run.py --list          # show the steps and exit
    python pipeline_run.py upload_csv      # run only the named step(s)
    python pipeline_run.py --continue-on-error   # keep going if a step fails
    python pipeline_run.py --from upload_csv     # start at a step, run the rest

Each step is launched as a separate process using the project's virtualenv
interpreter (.venv) when present, so its output streams straight to the console.
The script stops at the first failing step unless --continue-on-error is set.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (name, script, default extra args). `name` is what you type to select a step.
STEPS = [
    ("processor_data", "processor_data.py", ["--action", "all"]),
    ("upload_csv", "upload_csv.py", []),
    ("processor_documents", "processor_documents.py", []),
    ("processor_news", "processor_news.py", []),
    ("processor_press", "processor_press.py", []),
    ("upload_json", "upload_json.py", []),
]

# Named groups for scheduling: `--group csv` runs the market-data path,
# `--group json` runs the news/document/press path. Used by the Task Scheduler
# jobs so no wrapper .bat files are needed.
GROUPS = {
    "csv": ["processor_data", "upload_csv"],
    "json": ["processor_documents", "processor_news", "processor_press", "upload_json"],
}


def resolve_python() -> str:
    """Prefer the project's virtualenv interpreter, fall back to the current one."""
    candidates = [
        HERE / ".venv" / "Scripts" / "python.exe",  # Windows
        HERE / ".venv" / "bin" / "python",          # Linux/macOS
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def parse_args() -> argparse.Namespace:
    names = [name for name, _, _ in STEPS]
    parser = argparse.ArgumentParser(
        description="Run the data pipeline steps in order.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps (in order): " + " -> ".join(names),
    )
    parser.add_argument(
        "steps",
        nargs="*",
        metavar="STEP",
        help="Optional subset of steps to run (default: all). Choices: " + ", ".join(names),
    )
    parser.add_argument(
        "--group",
        choices=sorted(GROUPS),
        help="Run a named group of steps: " + "; ".join(f"{g}={'+'.join(s)}" for g, s in GROUPS.items()),
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Append all output to logs/<group|all>_YYYYMMDD.log instead of the console (for scheduled runs).",
    )
    parser.add_argument(
        "--from",
        dest="from_step",
        metavar="STEP",
        help="Start at this step and run every step after it.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep running the remaining steps even if one fails.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the steps in order and exit.",
    )
    return parser.parse_args()


def select_steps(args: argparse.Namespace) -> list:
    names = [name for name, _, _ in STEPS]

    if args.group:
        wanted = set(GROUPS[args.group])
        return [step for step in STEPS if step[0] in wanted]

    if args.steps:
        unknown = [s for s in args.steps if s not in names]
        if unknown:
            raise SystemExit(
                f"Unknown step(s): {', '.join(unknown)}. Choices: {', '.join(names)}"
            )
        # Preserve the canonical pipeline order regardless of how they were typed.
        return [step for step in STEPS if step[0] in set(args.steps)]

    if args.from_step:
        if args.from_step not in names:
            raise SystemExit(
                f"Unknown step for --from: {args.from_step}. Choices: {', '.join(names)}"
            )
        start = names.index(args.from_step)
        return STEPS[start:]

    return list(STEPS)


def main() -> int:
    args = parse_args()

    # --log: send everything (our prints + child output) to a dated file so the
    # scheduled tasks leave a trail without needing a wrapper .bat. Otherwise make
    # the console UTF-8 so Vietnamese/emoji lines from child scripts don't crash.
    log_file = None
    if args.log:
        logs_dir = HERE / "logs"
        logs_dir.mkdir(exist_ok=True)
        name = args.group or "all"
        log_file = open(logs_dir / f"{name}_{time.strftime('%Y%m%d')}.log", "a", encoding="utf-8")
        sys.stdout = log_file
        sys.stderr = log_file
        print("\n" + "#" * 70)
        print(f"# run {time.strftime('%Y-%m-%d %H:%M:%S')}  group={name}")
        print("#" * 70)
    else:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    if args.list:
        print("Pipeline steps (in order):")
        for i, (name, script, extra) in enumerate(STEPS, 1):
            suffix = (" " + " ".join(extra)) if extra else ""
            print(f"  {i}. {name:<20} -> {script}{suffix}")
        return 0

    python = resolve_python()
    steps = select_steps(args)

    print(f"Interpreter: {python}")
    print(f"Working dir: {HERE}")
    print(f"Running {len(steps)} step(s): " + " -> ".join(n for n, _, _ in steps))

    results = []
    pipeline_start = time.monotonic()

    for i, (name, script, extra) in enumerate(steps, 1):
        cmd = [python, str(HERE / script), *extra]
        print("\n" + "=" * 70)
        print(f"[{i}/{len(steps)}] {name}: {' '.join(cmd)}")
        print("=" * 70, flush=True)

        start = time.monotonic()
        completed = subprocess.run(
            cmd,
            cwd=str(HERE),
            stdout=log_file,
            stderr=subprocess.STDOUT if log_file else None,
        )
        elapsed = time.monotonic() - start
        ok = completed.returncode == 0
        results.append((name, ok, completed.returncode, elapsed))

        status = "OK" if ok else f"FAILED (exit {completed.returncode})"
        print(f"\n--> {name}: {status} in {elapsed:.1f}s", flush=True)

        if not ok and not args.continue_on_error:
            print(
                f"\nStopping: '{name}' failed. "
                "Fix the error above, or re-run with --continue-on-error / "
                f"--from {name} to resume.",
                file=sys.stderr,
            )
            break

    total = time.monotonic() - pipeline_start
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    for name, ok, code, elapsed in results:
        status = "OK    " if ok else f"FAIL {code}"
        print(f"  {status}  {name:<20} {elapsed:6.1f}s")
    print(f"Total: {total:.1f}s")

    ran = {name for name, _, _, _ in results}
    skipped = [name for name, _, _ in steps if name not in ran]
    if skipped:
        print("Skipped after failure: " + ", ".join(skipped))

    return 0 if all(ok for _, ok, _, _ in results) and not skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
