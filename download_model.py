#!/usr/bin/env python3
"""Download HuggingFace models (safetensors/pytorch) for vLLM.

Usage:
    ./download_model.py                                 interactive
    ./download_model.py <model-id>                      list available files
    ./download_model.py <model-id> --all                download all model files
    ./download_model.py <model-id> --all --bg           download in tmux session
    ./download_model.py --login                         save HuggingFace token
    ./download_model.py --status                        show active background downloads
    ./download_model.py --attach [session]              reattach to background download
"""

import sys, os, re, shutil, argparse, subprocess, shlex
from pathlib import Path
from datetime import datetime

def _c(code, t): return f"\033[{code}m{t}\033[0m"
def red(t):    return _c("0;31", t)
def green(t):  return _c("0;32", t)
def yellow(t): return _c("1;33", t)
def blue(t):   return _c("0;34", t)
def cyan(t):   return _c("0;36", t)

MODEL_DIR = Path("./llm_models/hf")

MODEL_FILE_PATTERNS = (
    ".safetensors", ".bin", ".pt", ".json", ".tiktoken",
    ".model", ".txt", ".index", ".pattern",
)

SKIP_PATTERNS = (
    ".gguf", ".onnx", ".msgpack", ".h5", ".safetensors.index.json",
)


def die(msg):
    print(red(f"Error: {msg}"), file=sys.stderr)
    sys.exit(1)


def format_size(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024: return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


TMUX_PREFIX = "hf-dl"


def _tmux_available() -> bool:
    try:
        subprocess.run(["tmux", "-V"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def _session_name(model_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "-", model_id)
    return f"{TMUX_PREFIX}-{safe}"


def _log_path(model_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "-", model_id)
    log_dir = Path("./llm_models/.download_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{safe}.log"


def _find_download_sessions() -> list[tuple[str, str]]:
    try:
        result = subprocess.run(
            ["tmux", "list-sessions", "-F", "#{session_name}\t#{session_created}"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            return []
        sessions = []
        for line in result.stdout.strip().splitlines():
            if not line.startswith(TMUX_PREFIX):
                continue
            parts = line.split("\t", 1)
            name = parts[0]
            created = datetime.fromtimestamp(int(parts[1])).strftime("%Y-%m-%d %H:%M") if len(parts) > 1 else "?"
            sessions.append((name, created))
        return sessions
    except Exception:
        return []


def _model_id_from_session(session_name: str) -> str:
    prefix = f"{TMUX_PREFIX}-"
    if session_name.startswith(prefix):
        return session_name[len(prefix):]
    return session_name


def cmd_status():
    sessions = _find_download_sessions()
    if not sessions:
        print(f"No active background downloads.")
        return
    print(f"Active background downloads:\n")
    for name, created in sessions:
        model_id = _model_id_from_session(name)
        log = _log_path(model_id)
        log_info = ""
        if log.exists():
            size = log.stat().st_size
            log_info = f"  log: {format_size(size)}"
        print(f"  {green(name)}  created: {created}{log_info}")
    print(f"\nAttach with:  {cyan('./download_model.py --attach <session-name>')}")
    print(f"View log:     {cyan('tail -f llm_models/.download_logs/<session>.log')}")


def cmd_attach(session_name: str | None):
    sessions = _find_download_sessions()
    if not sessions:
        die("No active background downloads to attach to.")
    if session_name:
        matches = [s for s in sessions if s[0] == session_name]
        if not matches:
            matches = [s for s in sessions if session_name in s[0]]
        if not matches:
            die(f"Session '{session_name}' not found. Active: {', '.join(s[0] for s in sessions)}")
        target = matches[0][0]
    else:
        if len(sessions) == 1:
            target = sessions[0][0]
        else:
            print("Multiple sessions found:")
            for name, created in sessions:
                print(f"  {green(name)}  created: {created}")
            print(f"\nSpecify one: {cyan('./download_model.py --attach <session-name>')}")
            return
    print(f"Attaching to {green(target)} — press Ctrl+B then D to detach.\n")
    os.execvp("tmux", ["tmux", "attach-session", "-t", target])


def _spawn_tmux_download(model_id: str):
    session = _session_name(model_id)
    log = _log_path(model_id)

    existing = _find_download_sessions()
    if any(s[0] == session for s in existing):
        die(f"Session '{session}' already exists. Attach with: ./download_model.py --attach {session}")

    script_path = Path(__file__).resolve()
    workdir = Path(__file__).parent.resolve()

    inner_cmd = (
        f'cd {shlex.quote(str(workdir))} && '
        f'PYTHONUNBUFFERED=1 python3 -u {shlex.quote(str(script_path))} --_exec-download --all {shlex.quote(model_id)} '
        f'2>&1 | tee {shlex.quote(str(log))}'
    )

    print(f"Starting download in tmux session: {green(session)}")
    print(f"Log file: {log}")
    print(f"\nCommands:")
    print(f"  Attach:  {cyan(f'./download_model.py --attach {session}')}")
    print(f"  Status:  {cyan('./download_model.py --status')}")
    print(f"  Log:     {cyan(f'tail -f {log}')}")
    print()

    try:
        subprocess.run(
            [
                "tmux", "new-session",
                "-d",
                "-s", session,
                "-x", "200", "-y", "50",
                "-c", str(workdir),
                "bash", "-c", inner_cmd,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        die(f"Failed to create tmux session: {e}")

    print(green(f"Download started in background."))
    print(f"  tmux attach -t {session}")


def _spawn_nohup_download(model_id: str):
    log = _log_path(model_id)
    script_path = Path(__file__).resolve()
    workdir = Path(__file__).parent.resolve()

    cmd = f'cd {shlex.quote(str(workdir))} && PYTHONUNBUFFERED=1 python3 -u {shlex.quote(str(script_path))} --_exec-download --all {shlex.quote(model_id)} > {shlex.quote(str(log))} 2>&1 &'

    print(f"Starting download with nohup (tmux not available).")
    print(f"Log file: {log}")
    print(f"  Monitor: {cyan(f'tail -f {log}')}")
    print()

    subprocess.Popen(
        ["nohup", "bash", "-c", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    print(green(f"Download started in background."))


def load_hf():
    try:
        import huggingface_hub as hf
        return hf
    except ImportError:
        die("huggingface_hub not installed.\n  Run: pip install huggingface_hub hf_transfer")


def enable_hf_transfer() -> bool:
    try:
        import hf_transfer  # noqa
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        xet_log = Path.home() / ".cache" / "huggingface" / "xet" / "logs"
        try:
            xet_log.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            pass

        if not os.access(xet_log, os.W_OK):
            print(
                f"{yellow('Warning:')} xet log directory not writable — this causes slow speeds.\n"
                f"  Fix once with:  sudo chown -R $USER ~/.cache/huggingface/"
            )

        os.environ.setdefault("RUST_LOG", "off")
        return True
    except ImportError:
        return False


def get_token(hf) -> str | None:
    t = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if t: return t
    try:
        t = hf.get_token()
        if t: return t
    except AttributeError:
        pass
    try:
        t = hf.HfFolder.get_token()
        if t: return t
    except Exception:
        pass
    return None


def save_token(hf, token: str):
    try:
        hf.login(token=token, add_to_git_credential=False)
    except Exception:
        hf.HfFolder.save_token(token)


def _errors(hf):
    names = ("RepositoryNotFoundError", "GatedRepoError", "EntryNotFoundError", "HfHubHTTPError")
    out = []
    for name in names:
        for mod in (hf, getattr(hf, "errors", None), getattr(hf, "utils", None)):
            cls = mod and getattr(mod, name, None)
            if cls:
                out.append(cls)
                break
        else:
            out.append(Exception)
    return tuple(out)


def is_model_file(path: str) -> bool:
    lower = path.lower()
    for skip in SKIP_PATTERNS:
        if lower.endswith(skip):
            return False
    for pat in MODEL_FILE_PATTERNS:
        if lower.endswith(pat):
            return True
    if "tokenizer" in lower or "config" in lower or "generation" in lower:
        return True
    if lower.endswith(".model") or lower.endswith(".txt"):
        return True
    return False


def list_model_files(hf, repo_id: str, token) -> list[tuple]:
    """Return [(repo_path, size_bytes, is_model_weight), ...]."""
    NotFound, Gated, _, HttpErr = _errors(hf)
    api = hf.HfApi(token=token)
    try:
        out = []
        for item in api.list_repo_tree(repo_id, repo_type="model", recursive=True):
            if not hasattr(item, "size"):
                continue
            path = item.path
            if not is_model_file(path):
                continue
            is_weight = any(path.endswith(ext) for ext in (".safetensors", ".bin", ".pt"))
            out.append((path, item.size or 0, is_weight))
        return out
    except Gated:
        die(
            f"{repo_id} is a gated model.\n"
            f"  Accept the license at: https://huggingface.co/{repo_id}\n"
            f"  Then set HF_TOKEN or run: ./download_model.py --login"
        )
    except NotFound:
        die(f"Model not found: {repo_id}")
    except HttpErr as e:
        die(str(e))


def download_model(hf, repo_id: str, token) -> Path:
    _, Gated, NotEntry, HttpErr = _errors(hf)

    org, model = repo_id.split("/", 1)
    out_dir = MODEL_DIR / org / model

    out_dir.mkdir(parents=True, exist_ok=True)

    existing_config = out_dir / "config.json"
    if existing_config.exists():
        try:
            ans = input(f"{yellow('Warning:')} {repo_id} already exists. Re-download? (y/N): ")
        except EOFError:
            ans = "n"
        if ans.strip().lower() != "y":
            print(green(f"Using existing: {out_dir}"))
            return out_dir
        shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    files = list_model_files(hf, repo_id, token)
    if not files:
        die(f"No model files found for {repo_id}")

    weights = [(p, s) for p, s, w in files if w]
    configs = [(p, s) for p, s, w in files if not w]

    total_size = sum(s for _, s, _ in files)
    print(f"\nDownloading {green(repo_id)}  ({len(weights)} weight files, {len(configs)} config files, {format_size(total_size)})\n")

    free = shutil.disk_usage(out_dir).free
    if free < total_size * 1.1:
        die(f"Not enough disk space — need {format_size(total_size)}, have {format_size(free)} free")

    try:
        for fpath, fsize, _ in files:
            print(f"  {green('↓')} {fpath}  ({format_size(fsize)})")

        hf.snapshot_download(
            repo_id=repo_id,
            local_dir=str(out_dir),
            token=token,
            allow_patterns=[p for p, _, _ in files],
        )

        print(green(f"✓  {out_dir}"))
        return out_dir

    except Gated:
        die(f"Access denied. Accept the license at https://huggingface.co/{repo_id}")
    except HttpErr as e:
        die(f"Download failed: {e}")
    except KeyboardInterrupt:
        print(f"\n{yellow('Interrupted.')} Partial files kept; re-run to resume.")
        sys.exit(0)


def main():
    ap = argparse.ArgumentParser(
        prog="download_model.py",
        description="Download HuggingFace models for vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s                                                 interactive
  %(prog)s Qwen/Qwen3-30B-A3B                              list files
  %(prog)s Qwen/Qwen3-30B-A3B --all                        download all
  %(prog)s Qwen/Qwen3-30B-A3B --all --bg                   background (tmux)
  %(prog)s --login                                         save HF token
  %(prog)s --status                                        show background downloads
  %(prog)s --attach [session]                              reattach to download
""",
    )
    ap.add_argument("model_id",     nargs="?", help="HuggingFace model ID (org/model)")
    ap.add_argument("--all",  action="store_true", help="download all model files")
    ap.add_argument("--list",  action="store_true", help="list files without downloading")
    ap.add_argument("--login", "--config", action="store_true",
                    help="save HuggingFace token for authenticated access")
    ap.add_argument("--bg",    action="store_true",
                    help="run download in a tmux session (SSH-safe)")
    ap.add_argument("--status", action="store_true",
                    help="list active background downloads")
    ap.add_argument("--attach", nargs="?", const="__auto__", default=None, metavar="SESSION",
                    help="attach to a background download session")
    ap.add_argument("--_exec-download", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.status:
        cmd_status()
        return

    if args.attach is not None:
        session = None if args.attach == "__auto__" else args.attach
        cmd_attach(session)
        return

    hf    = load_hf()
    fast  = enable_hf_transfer()
    token = get_token(hf)

    if args.login:
        print("Get a token at: https://huggingface.co/settings/tokens\n")
        try:
            t = input("Token: ").strip()
        except (KeyboardInterrupt, EOFError):
            print(); return
        if t:
            save_token(hf, t)
            print(green("Token saved."))
        else:
            print(yellow("No token entered."))
        return

    auth = green("authenticated") if token else yellow("no token — set HF_TOKEN for faster downloads")
    xfer = f"  {cyan('hf_transfer enabled')}" if fast else f"  {yellow('tip: pip install hf_transfer')}"
    bg_badge = f"  {cyan('[background mode]')}" if args.bg and not args._exec_download else ""
    print(f"{blue('HuggingFace Model Downloader')}  {auth}{xfer}{bg_badge}\n")

    model_id = args.model_id
    if not model_id:
        if args._exec_download:
            die("Internal error: --_exec-download requires model_id")
        if args.bg:
            die("--bg requires a model_id (no interactive mode in background)")
        try:
            model_id = input("Model ID (e.g. Qwen/Qwen3-30B-A3B): ").strip()
        except (KeyboardInterrupt, EOFError):
            print(); sys.exit(0)
    if "/" not in model_id:
        die("Model ID must be org/model — e.g. Qwen/Qwen3-30B-A3B")

    print(f"{blue('Info:')} Fetching file list...")
    files = list_model_files(hf, model_id, token)
    if not files:
        die(f"No model files found for {model_id}")

    weights = [(p, s) for p, s, w in files if w]
    configs = [(p, s) for p, s, w in files if not w]
    total = sum(s for _, s, _ in files)

    if args.list or not args.all:
        print(f"\nAvailable files for {green(model_id)}:\n")
        print(f"  {'Weight files:':}")
        for p, s in sorted(weights, key=lambda x: x[1], reverse=True):
            print(f"    {p}  ({format_size(s)})")
        if configs:
            print(f"\n  {'Config/other files:':}")
            for p, s in sorted(configs, key=lambda x: x[0]):
                print(f"    {p}  ({format_size(s)})")
        print(f"\n  Total: {format_size(total)} ({len(weights)} weights, {len(configs)} config)")
        if not args.all:
            print(f"\n  Download with: {cyan(f'./download_model.py {model_id} --all')}")
            return

    if args.bg and not args._exec_download:
        if _tmux_available():
            _spawn_tmux_download(model_id)
        else:
            _spawn_nohup_download(model_id)
        return

    download_model(hf, model_id, token)

    org, mdl = model_id.split("/", 1)
    print(f"\n{green('All done!')}  {MODEL_DIR}/{org}/{mdl}/")
    print(f"Run {cyan('./run_vllm_server.sh')} to start.")


if __name__ == "__main__":
    main()
