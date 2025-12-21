# Results Export Helpers
# =========================
RESULTS_FIELDS = [
    "run_id",
    "timestamp",
    "commit_sha",
    "config_hash",
    "model",
    "seed",
    "rounds",
    "broadcasts",
    "coverage",
    "correct_edges",
    "false_edges",
    "total_gt_edges",
    "mission_score",
    "norm_score",
    "runtime_s",
    "logfile"
    ]
try:
    _RESULTS_BASE = Path(__file__).resolve().parent
except NameError:
    _RESULTS_BASE = (Path("Code") / "corasat") if (Path("Code") / "corasat").exists() else Path.cwd()
RESULTS_PATH = _RESULTS_BASE / "results.csv"

def _safe_commit_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return None

def _config_hash_from_dict(cfg: Dict[str, Any]) -> Optional[str]:
    try:
        cfg_txt = json.dumps(cfg, sort_keys=True)
        return hashlib.sha1(cfg_txt.encode("utf-8")).hexdigest()[:10]
    except Exception:
        return None

def _compute_coverage_ratio(sim) -> Optional[float]:
    try:
        total = sim.grid_size[0] * sim.grid_size[1]
        if not total:
            return None
        visited = set()
        for drone in getattr(sim, "drones", []):
            for pos in getattr(drone, "mission_report", []):
                if isinstance(pos, (list, tuple)) and len(pos) == 2:
                    visited.add(tuple(pos))
        return round(len(visited) / total, 4)
    except Exception:
        return None

def _build_run_id(seed: Optional[Any]) -> str:
    base = datetime.now().strftime("%Y%m%d-%H%M%S")
    seed_part = str(seed) if seed is not None else "noseed"
    return f"{base}-{seed_part}-{uuid.uuid4().hex[:6]}"

def _append_results_rows(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

def persist_run_results(run_exports: List[Dict[str, Any]]) -> None:
    if not run_exports:
        return
    commit_sha = _safe_commit_sha()
    logfile = getattr(LOGGER, "log_path", None)
    rows: List[Dict[str, Any]] = []
    for entry in run_exports:
        sim = entry.get("sim")
        config = entry.get("config")
        seed = entry.get("seed")
        runtime_s = entry.get("runtime_s")
        timestamp = entry.get("timestamp") or datetime.now().isoformat()
        coverage = _compute_coverage_ratio(sim) if sim else None
        row = {
            "run_id": _build_run_id(seed),
            "timestamp": timestamp,
            "commit_sha": commit_sha,
            "config_hash": _config_hash_from_dict(config) if config else None,
            "model": getattr(sim, "model", None) if sim else None,
            "seed": seed,
            "rounds": getattr(sim, "round", None) if sim else None,
            "coverage": coverage,
            "correct_edges": getattr(sim, "correct_edge_counter", None) if sim else None,
            "false_edges": getattr(sim, "false_edge_counter", None) if sim else None,
            "total_gt_edges": len(getattr(sim, "gt_edges", []) or []),
            "broadcasts": getattr(sim, "broadcast_count", None) if sim else None,
            "mission_score": getattr(sim, "score", None) if sim else None,
            "norm_score": round(getattr(sim, "score", 0) / len(getattr(sim, "gt_edges", []) or []), 5) if sim and getattr(sim, "gt_edges", None) else None,
            "runtime_s": round(runtime_s, 2) if isinstance(runtime_s, (int, float)) else None,
            "logfile": logfile,
        }
        rows.append(row)
    _append_results_rows(rows)
    LOGGER.log(f"results.csv updated with {len(rows)} run(s).")


def _export_notebook_to_py(nb_path: str, out_py: str = "run_simulation.py") -> None:
    try:
        nb = nbformat.read(nb_path, as_version=4)
        body, _ = PythonExporter().from_notebook_node(nb)
        with open(out_py, "w", encoding="utf-8") as f:
            f.write(body)
        LOGGER.log(f"Exported notebook '{nb_path}' -> '{out_py}'")
    except Exception as e:
        LOGGER.log(f"ERROR: Notebook export failed: {e}")

try:
    if CONFIG["simulation"].get("create_py_export", True):
        if _running_in_notebook():
            nbp = _find_notebook_path()
            if nbp:
                _export_notebook_to_py(nbp, out_py="run_simulation.py")
except Exception:
    pass


# Timestamped Logger
class TimestampedLogger:
    def __init__(self, log_dir='logs', log_file='simulation.log'):
        date_tag = datetime.now().strftime('%Y-%m-%d')
        log_file = f'simulation_{date_tag}.log'
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = log_dir_path / log_file

        root = logging.getLogger()
        for h in list(root.handlers):
            try: h.close()
            except Exception: pass
            root.removeHandler(h)

        try:
            if os.path.exists(self.log_path):
                os.remove(self.log_path)
        except Exception:
            pass

        fh = logging.FileHandler(self.log_path, mode='a', encoding='utf-8', delay=False)
        ch = logging.StreamHandler()

        fmt = logging.Formatter(fmt='%(levelname)s:%(name)s:%(message)s')
        fh.setFormatter(fmt); ch.setFormatter(fmt)
        root.setLevel(logging.INFO)
        root.addHandler(fh); root.addHandler(ch)

        logging.getLogger('httpx').setLevel(logging.INFO)

        self.start_time = time.time()
        self.last_time = self.start_time
        self.log("Logger initialized.")

    def _now(self): return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _duration(self):
        current_time = time.time()
        d = current_time - self.last_time
        self.last_time = current_time
        return f"{d:.3f}s"

    def log(self, message):
        """Log a message with timestamp and duration since last log."""
        logging.info(f"[{self._now()}] (+{self._duration()}) {message}")
LOGGER = TimestampedLogger()

# Notebook Export
# =========================
def _running_in_notebook() -> bool:
    try:
        # Import get_ipython from the concrete module to satisfy static analyzers
        from IPython.core.getipython import get_ipython
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False

def _find_notebook_path() -> Optional[str]:
    try:
        import ipynbname
        return str(ipynbname.path())
    except Exception:
        pass
    try:
        candidates = [nb_path for nb_path in os.listdir(".") if nb_path.endswith(".ipynb")]
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
    except Exception:
        pass
    return None