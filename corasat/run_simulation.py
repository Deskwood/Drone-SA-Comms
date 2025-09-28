#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Sim Support

import logging
import os
import pprint
import time
import random
import re
from datetime import datetime
from typing import Tuple, List, Optional, Literal, Dict, Any, Set
import json
import pygame
import pyperclip
from concurrent.futures import ThreadPoolExecutor, Future
import nbformat
from nbconvert import PythonExporter
from pydantic import BaseModel
from pydantic import BaseModel, field_validator, model_validator
from typing import TypedDict, Optional, Literal, List, Dict, Any, Set, Tuple
try:
    from ollama import chat as ollama_chat
    _OLLAMA_AVAILABLE = True
except Exception:
    _OLLAMA_AVAILABLE = False

# ---------------- Constants ----------------
COLORS = ["white", "black"]
FIGURE_TYPES = ["king", "queen", "rook", "bishop", "knight", "pawn"]
DIRECTION_MAP: Dict[str, Tuple[int, int]] = {
    "north": (0, 1),
    "south": (0, -1),
    "east": (1, 0),
    "west": (-1, 0),
    "northeast": (1, 1),
    "northwest": (-1, 1),
    "southeast": (1, -1),
    "southwest": (-1, -1)
}
VALID_DIRECTIONS = set(DIRECTION_MAP.keys())
PLAN_PREFIX = "PLAN:"

class _Rec(TypedDict):
    action: str
    specifier: Optional[str]
    score: float
    rationale: str
    features: Dict[str, float]

class _DTNode(TypedDict, total=False):
    action: str
    specifier: Optional[str]
    score: float
    features: Dict[str, float]
    children: List["_DTNode"]

def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(max(0.0, v) for v in w.values())) or 1.0
    return {k: max(0.0, v) / s for k, v in w.items()}

def _sign(x: float) -> str:
    return "+" if x >= 0 else "-"

def _format_features(feats: Dict[str, float], weights: Dict[str, float]) -> str:
    parts = []
    for k in ["recall","plan_adherence","comm_opportunity","exploration","move_validity","precision"]:
        if k in feats:
            parts.append(f"{k}({_sign(feats[k])}{abs(feats[k]):.2f}×{weights.get(k,0):.2f})")
    return ", ".join(parts)

def _visited_token(x:int,y:int) -> str:
    return f"VISITED:{x},{y}"

def _compute_allowed_from(x:int, y:int, W:int, H:int) -> List[str]:
    out=[]
    for name,(dx,dy) in DIRECTION_MAP.items():
        nx, ny = x+dx, y+dy
        if 0 <= nx < W and 0 <= ny < H:
            out.append(name)
    return out

class _Rec(TypedDict):
    action: str
    specifier: Optional[str]
    score: float
    rationale: str
    features: Dict[str, float]

def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(max(0.0, v) for v in w.values())) or 1.0
    return {k: max(0.0, v) / s for k, v in w.items()}

def _visited_token(x:int,y:int) -> str: return f"VISITED:{x},{y}"

def _format_features(feats: Dict[str, float], weights: Dict[str, float]) -> str:
    items = []
    for k in ["recall","exploration","plan_adherence","comm_opportunity","move_validity","precision"]:
        if k in feats:
            items.append(f"{k}={feats[k]:+.2f}×{weights.get(k,0):.2f}")
    return ", ".join(items)

def _compute_allowed_from(x:int, y:int, W:int, H:int) -> List[str]:
    out=[]
    for name,(dx,dy) in DIRECTION_MAP.items():
        nx, ny = x+dx, y+dy
        if 0 <= nx < W and 0 <= ny < H:
            out.append(name)
    return out

# --- decision_support.score_action_sequence (replace function body) ---
def score_action_sequence(local_state: Dict[str, Any],
                          plan: List[str],
                          sequence: List[Tuple[str, Optional[str]]],
                          weights: Dict[str,float]) -> Tuple[float, Dict[str,float]]:
    W, H = CONFIG["board"]["width"], CONFIG["board"]["height"]
    feats = {"recall":0.0,"exploration":0.0,"plan_adherence":0.0,"move_validity":0.0,"comm_opportunity":0.0,"precision":0.0}
    x, y = local_state["pos"]
    allowed = set(local_state["allowed_dirs"])
    # tolerate either key
    same_ct = int(local_state.get("same_tile_count", local_state.get("same_tile_drones", 0)))
    neighbor_figs: Dict[str,str] = local_state.get("neighbor_figs", {})
    memory: str = local_state.get("memory","")
    next_plan = plan[0] if plan else None
    last_xy = (x, y)

    step = 0
    for (act, spec) in sequence:
        step += 1
        if act == "move":
            d = (spec or "").lower()
            legal = d in allowed
            feats["move_validity"] += 1.0 if legal else 0.0
            if next_plan:
                if d == next_plan and step == 1: feats["plan_adherence"] += 1.0
                elif d in plan[1:]: feats["plan_adherence"] += 0.5
                else: feats["plan_adherence"] -= 0.25
            if d in neighbor_figs:
                feats["recall"] += 1.0
                feats["precision"] += 0.05
            dx, dy = DIRECTION_MAP.get(d, (0,0))
            nx, ny = x+dx, y+dy
            vec = (nx - x, ny - y)
            mvdir = direction_from_vector(vec)
            if mvdir in neighbor_figs:
                feats["recall"] += 0.25  # small extra nudge
            if f"VISITED:{nx},{ny}" not in memory:
                feats["exploration"] += 1.0
            if (nx,ny) == last_xy:
                feats["exploration"] -= 0.5
            last_xy = (nx,ny)
            if 0 <= nx < W and 0 <= ny < H:
                x, y = nx, ny
                allowed = set(_compute_allowed_from(x,y,W,H))
            if same_ct > 0:
                feats["comm_opportunity"] -= 0.25
        elif act == "broadcast":
            feats["comm_opportunity"] += 1.0 if same_ct > 0 else 0.0
            if next_plan: feats["plan_adherence"] -= 0.1
        else:
            feats["comm_opportunity"] += 0.5 if same_ct > 0 else 0.0
            if next_plan: feats["plan_adherence"] -= 0.1

    w = _normalize_weights(weights)
    score = sum(feats[k]*w.get(k,0.0) for k in feats)
    return score, feats

def build_decision_tree(local_state: Dict[str, Any], plan: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    W, H = CONFIG["board"]["width"], CONFIG["board"]["height"]
    weights = cfg.get("weights", {})
    max_depth = int(cfg.get("max_depth", 2))
    max_branching = int(cfg.get("max_branching", 8))
    beam_width = int(cfg.get("beam_width", 8))
    timeout_ms = int(cfg.get("timeout_ms", 100))
    deterministic = bool(cfg.get("deterministic", True))

    def root_actions() -> List[Tuple[str, Optional[str]]]:
        acts: List[Tuple[str, Optional[str]]] = []
        for d in sorted(local_state.get("allowed_dirs", [])):
            acts.append(("move", d))
        acts.append(("broadcast", None))
        acts.append(("wait", None))
        return acts[:max_branching]

    # Evaluate root
    recs: List[_Rec] = []
    nodes: List[Dict[str, Any]] = []
    cand = root_actions()
    scored = []
    for a,s in cand:
        if (time.time()-start)*1000.0 > timeout_ms: break
        sc, feats = score_action_sequence(local_state, plan, [(a,s)], weights)
        scored.append((sc,a,s,feats))
    scored.sort(key=lambda t: (-t[0], t[1], t[2] or "")) if deterministic else scored.sort(key=lambda t: -t[0])
    scored = scored[:beam_width]

    for sc,a,s,feats in scored:
        node = {"action": a, "specifier": s, "score": float(sc), "features": feats, "children": []}
        # optional depth-2
        if max_depth >= 2 and (time.time()-start)*1000.0 <= timeout_ms:
            # simulate second step by reusing local_state scorer (pure heuristic)
            child_opts = [("wait",None),("broadcast",None)] + [("move", d) for d in sorted(local_state.get("allowed_dirs", []))]
            child_sc = []
            for a2,s2 in child_opts[:max_branching]:
                if (time.time()-start)*1000.0 > timeout_ms: break
                sc2, f2 = score_action_sequence(local_state, plan, [(a,s),(a2,s2)], weights)
                child_sc.append((sc2,a2,s2,f2))
            child_sc.sort(key=lambda t: (-t[0], t[1], t[2] or "")) if deterministic else child_sc.sort(key=lambda t: -t[0])
            for sc2,a2,s2,f2 in child_sc[:beam_width]:
                node["children"].append({"action": a2, "specifier": s2, "score": float(sc2), "features": f2})
        nodes.append(node)
        recs.append(_Rec(action=a, specifier=s, score=float(sc),
                         rationale=_format_features(feats, _normalize_weights(weights)),
                         features=feats))

    if deterministic:
        recs.sort(key=lambda r: (-r["score"], r["action"], r["specifier"] or ""))
    else:
        recs.sort(key=lambda r: -r["score"])

    return {"tree": {"action":"root","specifier":None,"children":nodes}, "recommendations": recs}

def format_decision_support_section(recs: List[_Rec], plan_next: Optional[str], k:int=3) -> str:
    lines = []
    lines.append("DecisionSupport:")
    lines.append(f"  PlanNext: {plan_next if plan_next else 'None'}")
    lines.append("  TopRecommendations:")
    for r in recs[:k]:
        spec = r["specifier"] if r["specifier"] else "-"
        lines.append(f"    - ({r['action']}, {spec}, {r['score']:.2f})")
    lines.append("  WhyTop:")
    if recs:
        for b in (recs[0]["rationale"] or "").split(", ")[:6]:
            lines.append(f"    - {b}")
    return "\n".join(lines[:30])

def build_decision_tree(local_state: Dict[str, Any], plan: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Returns {tree, recommendations}. Tree may be shallow if timeout triggers."""
    start = time.time()
    W, H = CONFIG["board"]["width"], CONFIG["board"]["height"]
    weights = cfg.get("weights", {})
    max_depth = int(cfg.get("max_depth", 2))
    max_branching = int(cfg.get("max_branching", 8))
    beam_width = int(cfg.get("beam_width", 8))
    timeout_ms = int(cfg.get("timeout_ms", 100))
    deterministic = bool(cfg.get("deterministic", True))

    def root_actions() -> List[Tuple[str, Optional[str]]]:
        acts: List[Tuple[str, Optional[str]]] = []
        # moves first in deterministic alphabetical order
        for d in sorted(local_state["allowed_dirs"]):
            acts.append(("move", d))
        acts.append(("broadcast", None))
        acts.append(("wait", None))
        return acts[:max_branching]

    def expand(x:int, y:int, allowed_dirs:Set[str], same_tile:int, depth:int) -> List[Tuple[str, Optional[str], int, int, Set[str]]]:
        """Generate children states for move/wait/broadcast."""
        out=[]
        # moves
        for d in sorted(allowed_dirs):
            dx, dy = DIRECTION_MAP[d]
            nx, ny = x+dx, y+dy
            child_allowed = set(_compute_allowed_from(nx,ny,W,H))
            out.append(("move", d, nx, ny, child_allowed))
            if len(out) >= max_branching: break
        # broadcast and wait keep position
        out.append(("broadcast", None, x, y, allowed_dirs))
        out.append(("wait", None, x, y, allowed_dirs))
        return out

    # Build shallow tree with beam pruning
    root_x, root_y = local_state["pos"]
    root_allowed = set(local_state["allowed_dirs"])
    root_children = []
    recs: List[_Rec] = []

    # Evaluate root actions
    candidates = root_actions()
    scored_root = []
    for act, spec in candidates:
        if (time.time() - start) * 1000.0 > timeout_ms: break
        s, feats = score_action_sequence(local_state, plan, [(act, spec)], weights)
        scored_root.append((s, act, spec, feats))
    # beam prune at root
    scored_root.sort(key=lambda t: (-t[0], t[1], t[2] or "")) if deterministic else scored_root.sort(key=lambda t: -t[0])
    scored_root = scored_root[:beam_width]

    # Optional deeper expansion (depth ≥ 2)
    for s0, act0, spec0, feats0 in scored_root:
        node: _DTNode = {"action": act0, "specifier": spec0, "score": float(s0), "features": feats0, "children": []}
        if max_depth >= 2 and (time.time() - start) * 1000.0 <= timeout_ms:
            # simulate state transition for depth-1 node
            x, y = root_x, root_y
            allowed = set(root_allowed)
            if act0 == "move":
                dx, dy = DIRECTION_MAP.get(spec0 or "", (0,0))
                x, y = x+dx, y+dy
                allowed = set(_compute_allowed_from(x,y,W,H))
            # build children one step deeper and keep best few
            child_opts = expand(x, y, allowed, local_state["same_tile_drones"], depth=2)
            child_scored=[]
            for act1, spec1, _, _, _ in child_opts:
                if (time.time() - start) * 1000.0 > timeout_ms: break
                s, feats = score_action_sequence(local_state, plan, [(act0, spec0), (act1, spec1)], weights)
                child_scored.append((s, act1, spec1, feats))
            child_scored.sort(key=lambda t: (-t[0], t[1], t[2] or "")) if deterministic else child_scored.sort(key=lambda t: -t[0])
            for s1, a1, sp1, f1 in child_scored[:beam_width]:
                node["children"].append({"action": a1, "specifier": sp1, "score": float(s1), "features": f1})
        root_children.append(node)

        # Build recommendation from root eval
        rationale = _format_features(feats0, _normalize_weights(weights))
        recs.append(_Rec(action=act0, specifier=spec0, score=float(s0), rationale=rationale, features=feats0))

    # Final ordering of recommendations
    recs.sort(key=lambda r: (-r["score"], r["action"], r["specifier"] or "")) if deterministic else recs.sort(key=lambda r: -r["score"])

    tree = {"action": "root", "specifier": None, "children": root_children}
    return {"tree": tree, "recommendations": recs}

def format_decision_support_section(recs: List[_Rec], plan_next: Optional[str], max_k:int=3) -> str:
    lines = []
    lines.append("DecisionSupport:")
    lines.append(f"  PlanNext: {plan_next if plan_next else 'None'}")
    lines.append("  TopRecommendations:")
    for r in recs[:max_k]:
        spec = r["specifier"] if r["specifier"] else "-"
        lines.append(f"    - ({r['action']}, {spec}, {r['score']:.2f})")
    lines.append("  WhyTop:")
    top = recs[0] if recs else None
    if top:
        # split features into short bullets
        bullets = (top["rationale"] or "").split(", ")
        for b in bullets[:6]:
            lines.append(f"    - {b}")
    return "\n".join(lines[:30])  # hard cap

# -------------- Optional: export notebook --------------
try:
    nb = nbformat.read("run_simulation.ipynb", as_version=4)
    body, _ = PythonExporter().from_notebook_node(nb)
    with open("run_simulation.py", "w", encoding="utf-8") as f:
        f.write(body)
except Exception:
    pass

# ---------------- Clean logger ----------------
class TimestampedLogger:
    def __init__(self, log_dir='logs', log_file='simulation.log'):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, log_file)

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

        fh = logging.FileHandler(self.log_path, mode='w', encoding='utf-8', delay=False)
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
        logging.info(f"[{self._now()}] (+{self._duration()}) {message}")

LOGGER = TimestampedLogger()

# ---------------- Config ----------------
def load_config(config_path: str = "config.json") -> dict:
    LOGGER.log(f"Load Config: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing config file: {config_path}")

    cfg.setdefault("prompt_requests", {})
    pr = cfg["prompt_requests"]
    pr.setdefault("schema",
        ("OUTPUT FORMAT: Return a SINGLE JSON object with keys "
         "rationale, action, direction, message, memory, found_edges. "
         "found_edges MUST be a JSON array (can be empty) of edges formatted as "
         "[[ [x1,y1],[x2,y2] ], ...]. Never omit found_edges. No extra text.")
    )

    cfg.setdefault("simulation", {})
    cfg.setdefault("board", {"width": 8, "height": 8})
    cfg.setdefault("gui", {
        "cell_size": 64,
        "margin": 2,
        "background_color": (30,30,30),
        "grid_color": (80,80,80),
        "drone_color": (200,200,50),
        "text_color": (20,20,20),
        "figure_image_dir": "figures",
        "sidebar_width": 480
    })
    sim = cfg["simulation"]
    sim.setdefault("max_rounds", 10)
    sim.setdefault("num_drones", 4)
    sim.setdefault("models", ["manual"])
    sim.setdefault("model_index", 0)
    sim.setdefault("temperature", 0.7)
    sim.setdefault("use_gui", True)
    sim.setdefault("headless", False)
    sim.setdefault("rules_path", cfg.get("rules_path", "rules.txt"))
    # generous budgets with cap
    sim.setdefault("max_tokens_for_rationale", 1024)
    sim.setdefault("max_tokens_for_action", 1024)
    sim.setdefault("max_tokens_for_action_move", 1024)
    sim.setdefault("max_tokens_for_action_broadcast", 1024)
    sim.setdefault("max_tokens_for_memory", 1024)
    sim.setdefault("max_tokens_total_cap", 4096)
    # planning + enforcement
    sim.setdefault("planning_rounds", 3)
    sim.setdefault("enforce_plan", True)
    # figure randomization
    sim.setdefault("randomize_figures", False)
    sim.setdefault("random_seed", None)

    # Decision support defaults
    cfg.setdefault("decision_support", {})
    ds = cfg["decision_support"]
    ds.setdefault("enabled", True)
    ds.setdefault("max_depth", 2)
    ds.setdefault("max_branching", 8)
    ds.setdefault("beam_width", 8)
    ds.setdefault("timeout_ms", 100)
    ds.setdefault("weights", {
        "recall": 0.5,
        "precision": 0.2,
        "plan_adherence": 0.2,
        "move_validity": 0.1,
        "comm_opportunity": 0.1,
        "exploration": 0.05
    })
    ds.setdefault("prefer_top_recommendation", False)
    ds.setdefault("include_in_prompt", True)
    ds.setdefault("deterministic", True)

    cfg.setdefault("figures", {c: {t: [] for t in FIGURE_TYPES} for c in COLORS})
    return cfg

CONFIG = load_config("config.json")

# ---------------- Images ----------------
def load_figure_images() -> dict:
    images = {}
    base_path = CONFIG["gui"]["figure_image_dir"]

    def try_load(path):
        return pygame.image.load(path) if os.path.exists(path) else None

    for color in COLORS:
        for figure_type in FIGURE_TYPES:
            candidates = [
                f"{color}{figure_type}.png",
                f"{color.capitalize()}{figure_type}.png",
                f"{color}{figure_type.capitalize()}.png",
                f"{color.capitalize()}{figure_type.capitalize()}.png"
            ]
            img = None
            for name in candidates:
                p = os.path.join(base_path, name)
                img = try_load(p)
                if img:
                    break
            if img:
                images[(color, figure_type)] = img
            else:
                LOGGER.log(f"Warning: Image not found for {color} {figure_type} in {base_path}")
    return images

FIGURE_IMAGES = {}

def direction_from_vector(vector: Tuple[int, int]) -> str:
    for direction, vec in DIRECTION_MAP.items():
        if vec == vector:
            return direction
    return str(vector)

# ---------------- TurnResult + JSON helpers ----------------
class TurnResult(BaseModel):
    rationale: str
    action: Literal["wait", "move", "broadcast"]
    direction: Optional[str] = None
    message: Optional[str] = None
    memory: str
    # Never None. Always a list (possibly empty) of [[x1,y1],[x2,y2]]
    found_edges: List[List[List[int]]] = []

    @model_validator(mode="after")
    def _require_specifiers(self) -> "TurnResult":
        if self.action == "move" and not isinstance(self.direction, str):
            raise ValueError("direction required when action=='move'")
        if self.action == "broadcast" and not isinstance(self.message, str):
            raise ValueError("message required when action=='broadcast'")
        return self

    @field_validator("found_edges", mode="before")
    @classmethod
    def _coerce_edges(cls, v):
        # Accept None → [], dict-form → convert, otherwise pass through
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return []

def _extract_first_json_block(text: str) -> str:
    start = text.find('{')
    if start == -1: return text
    stack = 0
    for i in range(start, len(text)):
        if text[i] == '{': stack += 1
        elif text[i] == '}':
            stack -= 1
            if stack == 0:
                return text[start:i+1]
    return text

def safe_parse_turnresult(payload: str) -> Dict[str, Any]:
    try:
        candidate = _extract_first_json_block(payload)
        data = json.loads(candidate)
        # Coerce found_edges early
        if data.get("found_edges") is None:
            data["found_edges"] = []
        return TurnResult.model_validate(data).model_dump()
    except Exception as e:
        # hard fallback with non-null found_edges
        return {
            "rationale": f"Parse/validate error: {e}",
            "action": "wait",
            "direction": None,
            "message": None,
            "memory": "",
            "found_edges": []
        }


def _normalize_edges(raw: Any) -> Set[Tuple[Tuple[int,int], Tuple[int,int]]]:
    """
    Accepts:
      - [[ [x1,y1], [x2,y2] ], ...]
      - or [{'src':[x1,y1], 'dst':[x2,y2]}, ...]
    Returns set({((x1,y1),(x2,y2)), ...}) with ints, drops invalid.
    """
    out: Set[Tuple[Tuple[int,int], Tuple[int,int]]] = set()
    if raw is None:
        return set()
    if not isinstance(raw, list):
        return out
    for item in raw:
        try:
            if isinstance(item, dict) and "src" in item and "dst" in item:
                a, b = item["src"], item["dst"]
            else:
                a, b = item
            x1, y1 = int(a[0]), int(a[1])
            x2, y2 = int(b[0]), int(b[1])
            out.add(((x1, y1), (x2, y2)))
        except Exception:
            continue
    return out

# ---------------- Plan parsing ----------------
PLAN_RE = re.compile(
    r'(?is)\bplan\b[^;:\n\r]*[:;]?\s*(?:v\d+;)?(?P<body>.*)$'
)
PATH_RE = re.compile(
    r'(?is)\b(?:d(?P<id>\d+)\s*:\s*)?path\s*=\s*(?P<seq>[a-z,\s]+)\b'
)

_ALIAS = {
    "n":"north","s":"south","e":"east","w":"west",
    "ne":"northeast","nw":"northwest","se":"southeast","sw":"southwest"
}
def _normalize_dirs(seq: str) -> List[str]:
    out = []
    for tok in (t.strip().lower() for t in seq.split(",") if t.strip()):
        tok = _ALIAS.get(tok, tok)
        if tok in DIRECTION_MAP:
            out.append(tok)
        else:
            LOGGER.log(f"Plan parser dropped token: {tok}")
    return out

def parse_plan_from_text(text: str, target_id: Optional[int] = None) -> List[str]:
    if not text: return []
    m = PLAN_RE.search(text) or PATH_RE.search(text)
    if not m:
        # also support plain "PATH=..." with no PLAN prefix
        m2 = PATH_RE.search(text)
        if not m2: return []
        body = m2.group(0)
        it = PATH_RE.finditer(body)
    else:
        body = m.group("body") if "body" in m.groupdict() else text
        it = PATH_RE.finditer(body)
    best: List[str] = []
    for pm in it:
        did = pm.group("id")
        seq = _normalize_dirs(pm.group("seq"))
        if target_id is None and seq:
            best = seq  # first found if no target filter
        elif did and target_id and int(did) == int(target_id) and seq:
            return seq
    return best

def update_plan_from_text(plans: Dict[int, List[str]], drone_id: int, text: str) -> bool:
    seq = parse_plan_from_text(text, target_id=drone_id)
    if seq:
        plans[drone_id] = seq
        return True
    return False

def _is_edge_locally_plausible(board, src: Tuple[int,int], dst: Tuple[int,int]) -> bool:
    x, y = src; tx, ty = dst
    W, H = CONFIG["board"]["width"], CONFIG["board"]["height"]
    if not (0 <= x < W and 0 <= y < H and 0 <= tx < W and 0 <= ty < H):
        return False
    here_tile = board[x][y]
    if not here_tile.figure:
        return False  # must stand on a figure
    # destination must be adjacent (local LOS) and occupied
    dx, dy = tx - x, ty - y
    if (dx, dy) not in DIRECTION_MAP.values():
        return False
    dst_tile = board[tx][ty]
    if not dst_tile.figure:
        return False

    f = here_tile.figure
    # legal-for-type, restricted to first step only
    if f.figure_type == "king":
        return True
    if f.figure_type == "pawn":
        # white captures up, black captures down
        caps = {(1,1),(-1,1)} if f.color == "white" else {(1,-1),(-1,-1)}
        return (dx,dy) in caps
    if f.figure_type == "rook":
        return (dx == 0) ^ (dy == 0) and abs(dx) <= 1 and abs(dy) <= 1
    if f.figure_type == "bishop":
        return abs(dx) == 1 and abs(dy) == 1
    if f.figure_type == "queen":
        lin = (dx == 0) ^ (dy == 0)
        diag = abs(dx) == 1 and abs(dy) == 1
        return (lin and abs(dx) <= 1 and abs(dy) <= 1) or diag
    # knight skipped for local adjacency
    return False

def candidate_edges_local(board: List[List["_Tile"]], pos: Tuple[int,int]) -> List[List[List[int]]]:
    x, y = pos
    tile = board[x][y]
    if not tile.figure:
        return []
    here = tile.figure
    W, H = CONFIG["board"]["width"], CONFIG["board"]["height"]

    def onb(a,b): return 0 <= a < W and 0 <= b < H
    out: List[List[List[int]]] = []

    if here.figure_type == "king":
        for dx,dy in DIRECTION_MAP.values():
            nx, ny = x+dx, y+dy
            if onb(nx,ny) and board[nx][ny].figure:
                out.append([[x,y],[nx,ny]])

    elif here.figure_type == "pawn":
        diags = [(1,1),(-1,1)] if here.color == "white" else [(1,-1),(-1,-1)]
        for dx,dy in diags:
            nx, ny = x+dx, y+dy
            if onb(nx,ny) and board[nx][ny].figure:
                out.append([[x,y],[nx,ny]])

    elif here.figure_type in ("rook","bishop","queen"):
        rays = []
        if here.figure_type in ("rook","queen"):
            rays += [(1,0),(-1,0),(0,1),(0,-1)]
        if here.figure_type in ("bishop","queen"):
            rays += [(1,1),(-1,-1),(1,-1),(-1,1)]
        # first step only (local)
        for dx,dy in rays:
            nx, ny = x+dx, y+dy
            if onb(nx,ny) and board[nx][ny].figure:
                out.append([[x,y],[nx,ny]])

    # knight skipped due to nonlocal jump
    return out


# ---------------- Randomize figures ----------------
def _randomize_figures_layout(figures_cfg: dict, board_w: int, board_h: int, seed: Optional[int] = None) -> dict:
    requests = []
    for color, types in figures_cfg.items():
        for ftype, positions in types.items():
            cnt = len(positions)
            if cnt > 0:
                requests.append((color, ftype, cnt))

    all_tiles = [(x, y) for x in range(board_w) for y in range(board_h)]
    rng = random.Random(seed)
    rng.shuffle(all_tiles)

    cursor = 0
    out = {c: {t: [] for t in FIGURE_TYPES} for c in COLORS}
    for color, ftype, cnt in requests:
        if cursor + cnt > len(all_tiles):
            raise ValueError("Not enough tiles to place all requested figures.")
        picks = all_tiles[cursor:cursor+cnt]
        cursor += cnt
        out[color][ftype] = [list(p) for p in picks]
    return out


# In[26]:


# Figures and Tiles

class _Figure:
    def __init__(self, position: Tuple[int, int], color: str, figure_type: str):
        self.position = position
        self.color = color
        self.figure_type = figure_type
        self.defended_by = 0
        self.attacked_by = 0
        self.target_positions: List[Tuple[int, int]] = []

    def calculate_figure_targets(self, board: List[List['_Tile']]):
        self.target_positions = []
        W, H = CONFIG["board"]["width"], CONFIG["board"]["height"]

        def on_board(x, y): return 0 <= x < W and 0 <= y < H

        if self.figure_type in ("queen", "rook", "bishop"):
            if self.figure_type == "rook":
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            elif self.figure_type == "bishop":
                directions = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
            else:  # queen
                directions = [(1, 0), (-1, 0), (0, 1), (0, -1),
                              (1, 1), (-1, -1), (1, -1), (-1, 1)]
            for dx, dy in directions:
                x, y = self.position
                while True:
                    x += dx; y += dy
                    if not on_board(x, y): break
                    self.target_positions.append((x, y))
                    if board[x][y].figure is not None:
                        break

        elif self.figure_type == "knight":
            for dx, dy in [(2, 1), (2, -1), (-2, 1), (-2, -1),
                           (1, 2), (1, -2), (-1, 2), (-1, -2)]:
                x = self.position[0] + dx
                y = self.position[1] + dy
                if on_board(x, y):
                    self.target_positions.append((x, y))

        elif self.figure_type == "king":
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (-1, -1), (1, -1), (-1, 1)]:
                x = self.position[0] + dx
                y = self.position[1] + dy
                if on_board(x, y):
                    self.target_positions.append((x, y))

        elif self.figure_type == "pawn":
            diagonals = [(1, 1), (-1, 1)] if self.color == "white" else [(1, -1), (-1, -1)]
            for dx, dy in diagonals:
                x = self.position[0] + dx
                y = self.position[1] + dy
                if on_board(x, y):
                    self.target_positions.append((x, y))

class _Tile:
    def __init__(self, x: int, y: int):
        self.x = x; self.y = y
        self.targeted_by = {"white": 0, "black": 0}
        self.figure: Optional[_Figure] = None
        self.drones: List['_Drone'] = []

    def set_figure(self, figure: _Figure): self.figure = figure
    def add_drone(self, drone: '_Drone'):
        if drone not in self.drones:
            self.drones.append(drone)
    def remove_drone(self, drone: '_Drone'):
        if drone in self.drones:
            self.drones.remove(drone)
    def reset_targeted_by_amounts(self): self.targeted_by = {"white": 0, "black": 0}
    def add_targeted_by_amount(self, color: str, amount: int = 1): self.targeted_by[color] += amount

def _compute_edges_for(figures: List[_Figure], board: List[List[_Tile]]) -> Set[Tuple[Tuple[int,int], Tuple[int,int]]]:
    edges: Set[Tuple[Tuple[int,int], Tuple[int,int]]] = set()
    for f in figures:
        for (tx, ty) in f.target_positions:
            if board[tx][ty].figure is not None:
                edges.add((f.position, board[tx][ty].figure.position))
    return edges


# In[ ]:


# Drones

class _Drone:
    def __init__(self, id: int, position: Tuple[int, int], model: str, rules: str, sim, color: str = "white"):
        self.id = id
        self.position = position
        self.color = color
        self.model = model
        self.sim = sim
        self.rules = rules.replace("DRONE_ID", str(self.id))\
                          .replace("NUMBER_OF_DRONES", str(CONFIG["simulation"]["num_drones"]))\
                          .replace("NUMBER_OF_ROUNDS", str(CONFIG["simulation"]["max_rounds"]))
        self.memory = ""     # per-drone memory (LLM-owned string)
        self.rx_buffer = ""  # per-drone inbox; only filled by co-located broadcasts
        # self.planned_path: List[str] = []  # parsed from plan broadcast during planning phase

    def _move(self, direction: str) -> bool:
        direction = (direction or "").lower()
        if direction in DIRECTION_MAP:
            dx, dy = DIRECTION_MAP[direction]
            nx, ny = self.position[0] + dx, self.position[1] + dy
            if 0 <= nx < CONFIG["board"]["width"] and 0 <= ny < CONFIG["board"]["height"]:
                self.sim.board[self.position[0]][self.position[1]].remove_drone(self)
                self.position = (nx, ny)
                self.sim.board[nx][ny].add_drone(self)
                LOGGER.log(f"Drone {self.id} moved to {self.position}.")
                return True
            else:
                LOGGER.log(f"Drone {self.id} attempted OOB move to {(nx,ny)}.")
        else:
            LOGGER.log(f"Drone {self.id} attempted invalid direction '{direction}'.")
        return False

    def _allowed_directions(self) -> List[str]:
        x, y = self.position
        W, H = CONFIG["board"]["width"], CONFIG["board"]["height"]
        allowed = []
        for name, (dx, dy) in DIRECTION_MAP.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H:
                allowed.append(name)
        return allowed

    def _phase(self) -> str:
        return "Planning" if self.sim.round <= self.sim.planning_rounds else "Execution"

    def _determine_situation_description(self) -> str:
        # Only local, directly observable info + own memory + received broadcasts
        same_tile_drones = [
            f"Drone {d.id}" for d in self.sim.board[self.position[0]][self.position[1]].drones if d.id != self.id
        ]

        fig_here = "None"
        if self.sim.board[self.position[0]][self.position[1]].figure:
            fig_here = self.sim.board[self.position[0]][self.position[1]].figure.figure_type

        neigh = ""
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx==0 and dy==0:
                    continue
                nx, ny = self.position[0]+dx, self.position[1]+dy
                if 0 <= nx < CONFIG["board"]["width"] and 0 <= ny < CONFIG["board"]["height"]:
                    t = self.sim.board[nx][ny]
                    if t.figure:
                        neigh += f"{direction_from_vector((dx,dy))}: {t.figure.color}, "
        neigh = neigh.strip(", ")

        allowed = self._allowed_directions()

        s = []
        s.append(f"Phase: {self._phase()}")
        s.append(f"Current round number: {self.sim.round}")
        s.append(f"Board size: {CONFIG['board']['width']}x{CONFIG['board']['height']} (x=0..{CONFIG['board']['width']-1}, y=0..{CONFIG['board']['height']-1})")
        s.append(f"My grid coords: x={self.position[0]}, y={self.position[1]}")
        s.append(f"Current position: {self.position}")
        s.append(f"AllowedDirections: {allowed}")
        s.append("Reminder: You MUST pick 'direction' only from AllowedDirections when action=='move'.")
        s.append(f"Visible drones at position: {', '.join(same_tile_drones) if same_tile_drones else 'None'}")
        s.append(f"Visible figure at position: {fig_here}")
        s.append(f"Visible neighboring figures: {neigh or 'None'}")
        s.append(f"Memory: {self.memory}")
        s.append(f"Broadcast Rx Buffer: {self.rx_buffer}")
        self.rx_buffer = ""  # drain the inbox each turn
        return "\n".join(s)

    # LLM interface (unchanged besides using only local situation above)
    def _token_budget_total(self) -> int:
        sim = CONFIG["simulation"]
        est = (int(sim.get("max_tokens_for_rationale",2048)) +
               int(sim.get("max_tokens_for_action",2048)) +
               max(int(sim.get("max_tokens_for_action_move",2048)),
                   int(sim.get("max_tokens_for_action_broadcast",2048))) +
               int(sim.get("max_tokens_for_memory",2048)))
        cap = int(sim.get("max_tokens_total_cap", 4096))
        return max(512, min(est, cap))

    def _generate_single_model_response(self, messages: List[dict], model: str, temperature: float) -> List[dict]:
        def _store(text: str) -> List[dict]:
            validated = safe_parse_turnresult(text)
            messages.append({"role": "assistant", "content": validated})
            return messages

        max_tokens_total = self._token_budget_total()
        num_predict = max(1024, max_tokens_total)

        if model == "manual":
            try: pyperclip.copy(messages[-1]["content"])
            except Exception: pass
            return _store(input("Paste pure JSON TurnResult: "))

        if not _OLLAMA_AVAILABLE:
            return _store(json.dumps({"rationale": "Ollama not installed; wait.", "action": "wait", "direction": None, "message": None, "memory": ""}))

        def _ollama(extra_hint: Optional[str] = None, np: int = num_predict):
            mm = messages if not extra_hint else messages + [{"role": "user", "content": extra_hint}]
            resp = ollama_chat(model=model, messages=mm, stream=False, format="json",
                               options={"temperature": float(temperature), "num_predict": int(np)})
            content = getattr(resp, "message", None)
            if content and hasattr(content, "content"):
                return content.content or ""
            if isinstance(resp, dict):
                return resp.get("message", {}).get("content", "")
            return str(resp)

        raw = _ollama()
        parsed = safe_parse_turnresult(raw)
        # Retry if parse failed OR found_edges is missing
        if (parsed.get("action") == "wait" and str(parsed.get("rationale","")).startswith("Parse/validate error")) \
            or ("found_edges" not in parsed):
            raw2 = _ollama("REMINDER: Output ONLY a single valid JSON object exactly matching the schema. No prose.",
                            np=int(num_predict * 2))
            # final attempt
            parsed2 = safe_parse_turnresult(raw2)
            if "found_edges" not in parsed2:
                # force an empty list to keep scoring consistent
                try:
                    d = json.loads(_extract_first_json_block(raw2))
                    d["found_edges"] = []
                    return _store(json.dumps(d))
                except Exception:
                    return _store(json.dumps({"rationale": "Schema still missing; waiting.", "action": "wait", "direction": None, "message": None, "memory": "", "found_edges": []}))
            return _store(raw2)
        return _store(raw)

    def _local_state(self) -> Dict[str, Any]:
        x, y = self.position
        tile = self.sim.board[x][y]
        same_ct = sum(1 for d in tile.drones if d.id != self.id)
        neighbor_figs: Dict[str,str] = {}
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx==0 and dy==0: continue
                nx, ny = x+dx, y+dy
                if 0 <= nx < CONFIG["board"]["width"] and 0 <= ny < CONFIG["board"]["height"]:
                    t = self.sim.board[nx][ny]
                    if t.figure:
                        neighbor_figs[direction_from_vector((dx,dy))] = t.figure.color
        # provide BOTH keys for compatibility
        return {
            "pos": (x,y),
            "allowed_dirs": self._allowed_directions(),
            "same_tile_count": same_ct,
            "same_tile_drones": same_ct,   # <= legacy key
            "fig_here": tile.figure.figure_type if tile.figure else None,
            "neighbor_figs": neighbor_figs,
            "memory": self.memory or "",
            "rx_preview": self.rx_buffer or ""
        }

    def generate_full_model_response(self) -> List[dict]:
        temperature = CONFIG["simulation"].get("temperature", 0.7)

        # Build base SITUATION and clear rx
        situation = self._determine_situation_description()

        # Decision support block
        ds_cfg = CONFIG.get("decision_support", {"enabled": False})
        if ds_cfg.get("enabled", False):
            local_state = self._local_state()
            plan_queue = self.sim.plans.get(self.id, [])
            ds_result = build_decision_tree(local_state, plan_queue, ds_cfg)
            recs = ds_result.get("recommendations", [])
            # store top for gating
            if recs:
                self.sim._ds_top_by_drone[self.id] = {"action": recs[0]["action"], "specifier": recs[0]["specifier"]}
            else:
                self.sim._ds_top_by_drone[self.id] = None
            if ds_cfg.get("include_in_prompt", True) and recs:
                situation += "\n" + format_decision_support_section(recs, self.sim._next_planned_step(self.id), 3)

        # SuggestedEdges block (local only)
        se = candidate_edges_local(self.sim.board, self.position)
        if se:
            situation += "\nSuggestedEdges: " + json.dumps(se)
        strict_edge_hint = (
        "EDGE RULE: Output edges ONLY from SuggestedEdges. "
        "Do not invent edges. Keep found_edges a list. If SuggestedEdges is empty, use []."
        )
        situation += "\n" + strict_edge_hint
        # Append cues
        pr = CONFIG.get("prompt_requests", {})
        cues = "\n".join([
            pr.get("schema",""),
            pr.get("rationale",""),
            pr.get("action",""),
            pr.get("action_move",""),
            pr.get("action_broadcast",""),
            pr.get("memory_update","")
        ]).strip()
        user_content = situation if not cues else situation + "\n\n" + cues

        messages = [
            {"role": "system", "content": self.rules},
            {"role": "user", "content": user_content}
        ]
        print(f"Context length: {len(user_content)+len(self.rules)} chars")
        return self._generate_single_model_response(messages=messages, model=self.model, temperature=temperature)


# In[28]:


# GUI

class _SimulationGUI:
    def __init__(self, sim):
        self.sim = sim
        self.grid_size = (CONFIG["board"]["width"], CONFIG["board"]["height"])

        pygame.init()
        gui = CONFIG["gui"]
        self.sidebar_width = gui.get("sidebar_width", 480)
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF

        board_w = self.grid_size[0]*(gui["cell_size"]+gui["margin"]) + gui["margin"]
        board_h = self.grid_size[1]*(gui["cell_size"]+gui["margin"]) + gui["margin"]
        total_w = board_w + self.sidebar_width
        total_h = board_h

        self.screen = pygame.display.set_mode((total_w, total_h), flags)
        pygame.display.set_caption(
            f'Simulation - Round {1}.1/{CONFIG["simulation"]["max_rounds"]}.{CONFIG["simulation"]["num_drones"]}'
        )
        self.clock = pygame.time.Clock()

        global FIGURE_IMAGES
        if not FIGURE_IMAGES:
            FIGURE_IMAGES = load_figure_images()
        for k, surf in list(FIGURE_IMAGES.items()):
            try: FIGURE_IMAGES[k] = surf.convert_alpha()
            except pygame.error: pass

        self._image_cache = {}
        try:
            self._font = pygame.font.SysFont(None, 18)
            self._font_small = pygame.font.SysFont(None, 16)
        except Exception:
            pygame.font.init()
            self._font = pygame.font.Font(None, 18)
            self._font_small = pygame.font.Font(None, 16)

        self.info_lines: List[str] = []
        self.info_max_lines = 400
        self.info_scroll = 0

    def _get_scaled(self, img):
        key = (id(img), CONFIG["gui"]["cell_size"])
        if key not in self._image_cache:
            self._image_cache[key] = pygame.transform.scale(img, (CONFIG["gui"]["cell_size"], CONFIG["gui"]["cell_size"]))
        return self._image_cache[key]

    def _draw_score_panel(self, x0: int, y0: int, w: int):
        s = self.screen
        pad = 12
        y = y0

        s.blit(self._font.render("Score", True, (220, 220, 220)), (x0 + pad, y)); y += 24
        stats = self.sim.score_stats()
        items = [
            ("Phase", self.sim.phase_label()),
            ("Identified nodes", stats["identified_nodes"]),
            ("Discovered edges", stats["discovered_edges"]),
            ("GT edges", stats["gt_edges"]),
            ("Correct edges", stats["correct_edges"]),
            ("False edges", stats["false_edges"]),
            ("Score", stats["score"]),
            ("Precision", f'{stats["precision"]:.2f}'),
            ("Recall", f'{stats["recall"]:.2f}')
        ]
        for lbl, val in items:
            s.blit(self._font_small.render(f"{lbl}: {val}", True, (200, 200, 200)), (x0 + pad, y))
            y += 18

        y += 6
        s.blit(self._font.render("Plans", True, (220, 220, 220)), (x0 + pad, y)); y += 20
        for d in self.sim.drones:
            nxt = self.sim._next_planned_step(d.id)
            q = self.sim.plans.get(d.id, [])
            preview = f"next={nxt or '-'} | queue={q}" if q else "next=- | queue=[]"
            s.blit(self._font_small.render(f"Drone {d.id}: {preview}", True, (200, 200, 200)), (x0 + pad, y))
            y += 18

        y += 8
        s.blit(self._font_small.render("Latest at bottom", True, (160, 160, 160)), (x0 + w - 150, y0 + 2))
        pygame.draw.line(s, (60,60,60), (x0, y), (x0 + w, y), 1)
        return y + 10

    def _draw_sidebar(self):
        gui = CONFIG["gui"]
        s = self.screen

        board_w = self.grid_size[0]*(gui["cell_size"]+gui["margin"]) + gui["margin"]
        x0 = board_w
        y0 = 0
        w = self.sidebar_width
        h = s.get_height()

        pygame.draw.rect(s, (25,25,25), pygame.Rect(x0, y0, w, h))
        pygame.draw.line(s, (60,60,60), (x0, 0), (x0, h), 1)

        y_log_top = self._draw_score_panel(x0, y0 + 8, w)

        pad = 12
        tx = x0 + pad
        ty = y_log_top + 8
        tw = w - 2 * pad
        th = h - ty - pad

        pygame.draw.rect(s, (32,32,32), pygame.Rect(tx-4, ty-4, tw+8, th+8))

        line_h = self._font_small.get_height() + 4
        max_lines_fit = th // line_h
        start_idx = max(0, len(self.info_lines) - max_lines_fit - self.info_scroll)
        end_idx = len(self.info_lines) - self.info_scroll

        y = ty
        for line in self.info_lines[start_idx:end_idx]:
            if y > ty + th - line_h:
                break
            s.blit(self._font_small.render(line, True, (230,230,230)), (tx, y))
            y += line_h

    def post_info(self, text: str):
        for line in text.splitlines():
            self.info_lines.append(line)
        if len(self.info_lines) > self.info_max_lines:
            self.info_lines = self.info_lines[-self.info_max_lines:]
        self.info_scroll = 0

    def draw_field(self):
        gui = CONFIG["gui"]
        cell = gui["cell_size"]
        m = gui["margin"]
        gw, gh = self.grid_size

        self.screen.fill(gui["background_color"])

        for x in range(gw):
            for y in range(gh):
                y_flip = gh - 1 - y
                rect = pygame.Rect(x*(cell+m)+m, y_flip*(cell+m)+m, cell, cell)
                pygame.draw.rect(self.screen, gui["grid_color"], rect)

                tile = self.sim.board[x][y]

                if tile.figure:
                    img = FIGURE_IMAGES.get((tile.figure.color, tile.figure.figure_type))
                    if img:
                        self.screen.blit(self._get_scaled(img), rect.topleft)
                    else:
                        pygame.draw.circle(self.screen, (200,200,200), rect.center, cell//3)

                if tile.figure:
                    fig = tile.figure
                    overlay = f"D{fig.defended_by} A{fig.attacked_by}"
                    surf = self._font.render(overlay, True, gui["text_color"])
                    pad = 3
                    tx = rect.left + pad
                    ty = rect.bottom - surf.get_height() - pad
                    pygame.draw.rect(self.screen, (40,40,40), pygame.Rect(tx-2, ty-1, surf.get_width()+4, surf.get_height()+2))
                    self.screen.blit(surf, (tx, ty))

                total = len(tile.drones)
                if total > 0:
                    angle_step = 360/total if total > 1 else 0
                    radius = cell//6
                    for d_idx, drone in enumerate(tile.drones):
                        offset = pygame.math.Vector2(0,0)
                        if total > 1:
                            offset = pygame.math.Vector2(1,0).rotate(angle_step*d_idx) * (cell//4)
                        center = (rect.centerx + int(offset.x), rect.centery + int(offset.y))
                        pygame.draw.circle(self.screen, gui["drone_color"], center, radius)
                        t = self._font.render(str(drone.id), True, gui["text_color"])
                        self.screen.blit(t, t.get_rect(center=center))

        if self.sim._thinking:
            cur = self.sim.current_drone()
            if cur:
                x, y = cur.position
                y_flip = gh - 1 - y
                rect = pygame.Rect(x*(cell+m)+m, y_flip*(cell+m)+m, cell, cell)
                pygame.draw.rect(self.screen, (255,215,0), rect, 2)

        self._draw_sidebar()
        pygame.display.flip()


# In[ ]:


# Simulation

class Simulation:
    def __init__(self):
        if CONFIG["simulation"].get("headless", False):
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

        self.turn = 1
        self.round = 1
        rules_path = CONFIG["simulation"].get("rules_path", CONFIG.get("rules_path", "rules.txt"))
        with open(rules_path, "r", encoding="utf-8") as f:
            self.rules = f.read().replace("NUMBER_OF_DRONES", str(CONFIG["simulation"]["num_drones"]))

        self.grid_size = (CONFIG["board"]["width"], CONFIG["board"]["height"])
        self.max_rounds = CONFIG["simulation"]["max_rounds"]
        self.num_drones = CONFIG["simulation"]["num_drones"]
        self.models = CONFIG["simulation"]["models"]
        self.model_index = CONFIG["simulation"]["model_index"]
        self.model = self.models[self.model_index]
        LOGGER.log(f"Using model: {self.model}")

        self.planning_rounds = int(CONFIG["simulation"].get("planning_rounds", 3))
        self.enforce_plan = bool(CONFIG["simulation"].get("enforce_plan", True))
        self.plans: Dict[int, List[str]] = {}  # optional plan-queue if drones put PLAN:path=... in memory

        # World
        self.board = [[_Tile(x, y) for y in range(self.grid_size[1])] for x in range(self.grid_size[0])]
        self.figures: List[_Figure] = []
        self.drones: List[_Drone] = []

        # Scoring (team-level truth; does not leak to drones)
        self.gt_edges: Set[Tuple[Tuple[int,int], Tuple[int,int]]] = set()
        # Drones report edges they believe exist; we aggregate
        self.drone_edges: Dict[int, Set[Tuple[Tuple[int,int], Tuple[int,int]]]] = {}

        self._edge_log_seen: Set[Tuple[Tuple[int,int], Tuple[int,int]]] = set()

        self._create_figures()
        self._rebuild_threat_map()
        self._compute_attack_defense_per_figure()
        self._compute_gt_edges()

        if self.figures:
            wk = next((f for f in self.figures if f.color == "white" and f.figure_type == "king"), self.figures[0])
            self.drone_base = wk.position
        else:
            self.drone_base = (0, 0)
        self._create_drones()

        if CONFIG["simulation"]["use_gui"]:
            self.gui = _SimulationGUI(self)

        self.executor = ThreadPoolExecutor(max_workers=1)
        self._current_future: Optional[Future] = None
        self._thinking = False
        self._ds_top_by_drone: Dict[int, Optional[Dict[str, Any]]] = {}


    # ---- Phase helpers
    def phase_label(self) -> str:
        return "Planning" if self.round <= self.planning_rounds else "Execution"

    # ---- Figures & edges (unchanged)
    def _create_figures(self):
        LOGGER.log("Creating figures based on configuration.")
        self.figures = []
        figures_cfg = CONFIG.get("figures", {})
        sim_cfg = CONFIG.get("simulation", {})
        W, H = self.grid_size

        rand_flag = bool(sim_cfg.get("randomize_figures", False))
        seed_val = sim_cfg.get("random_seed", None)
        should_randomize = rand_flag or (seed_val is not None)

        if should_randomize:
            try:
                seed = None
                if seed_val is not None:
                    s = str(seed_val).strip()
                    if s.lstrip("-").isdigit():
                        seed = int(s)
                randomized = _randomize_figures_layout(figures_cfg, W, H, seed)
                figures_cfg = randomized
                LOGGER.log(f"Figure positions RANDOMIZED (seed={seed if seed is not None else '<none>'}).")
                sample_lines = []
                for color in ("white", "black"):
                    for ftype in ("king", "queen", "rook", "bishop", "knight", "pawn"):
                        pos_list = randomized.get(color, {}).get(ftype, [])
                        if pos_list:
                            sample_lines.append(f"{color} {ftype}: {pos_list[:3]}{'...' if len(pos_list)>3 else ''}")
                if sample_lines:
                    LOGGER.log("Randomized sample: " + " | ".join(sample_lines))
            except Exception as e:
                LOGGER.log(f"Randomization failed ({e}); FALLING BACK to configured positions.")

        for color in COLORS:
            for figure_type in FIGURE_TYPES:
                for position in figures_cfg.get(color, {}).get(figure_type, []):
                    self.figures.append(_Figure(tuple(position), color, figure_type))

        for f in self.figures:
            self.board[f.position[0]][f.position[1]].set_figure(f)

    def _create_drones(self):
        LOGGER.log(f"Creating {self.num_drones} drones.")
        for i in range(self.num_drones):
            d = _Drone(id=i+1, position=self.drone_base, model=self.model, rules=self.rules, sim=self)
            vx, vy = d.position
            d.memory = f"VISITED:{vx},{vy}"
            self.drones.append(d)
        base = self.board[self.drone_base[0]][self.drone_base[1]]
        for d in self.drones:
            base.add_drone(d)

    def _rebuild_threat_map(self):
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                self.board[x][y].reset_targeted_by_amounts()
        for f in self.figures:
            f.calculate_figure_targets(self.board)
        for f in self.figures:
            for (tx, ty) in f.target_positions:
                self.board[tx][ty].add_targeted_by_amount(f.color, 1)

    def _compute_attack_defense_per_figure(self):
        for f in self.figures:
            tile = self.board[f.position[0]][f.position[1]]
            if f.color == "white":
                f.defended_by = tile.targeted_by["white"]
                f.attacked_by = tile.targeted_by["black"]
            else:
                f.defended_by = tile.targeted_by["black"]
                f.attacked_by = tile.targeted_by["white"]

    def _compute_gt_edges(self):
        self._rebuild_threat_map()
        self.gt_edges = _compute_edges_for(self.figures, self.board)
        LOGGER.log(f"GT Edges computed: {len(self.gt_edges)}")

    # ---- Union of reported edges
    def discovered_edges(self) -> Set[Tuple[Tuple[int,int], Tuple[int,int]]]:
        all_sets = self.drone_edges.values()
        out: Set[Tuple[Tuple[int,int], Tuple[int,int]]] = set()
        for s in all_sets:
            out |= s
        return out

    def score_stats(self) -> Dict[str, Any]:
        disc = self.discovered_edges()
        gt = self.gt_edges
        correct = disc & gt
        false = disc - gt
        prec = (len(correct) / len(disc)) if disc else 0.0
        rec = (len(correct) / len(gt)) if gt else 0.0
        nodes = set()
        for (a,b) in disc:
            nodes.add(a); nodes.add(b)
        return {
            "identified_nodes": len(nodes),
            "discovered_edges": len(disc),
            "gt_edges": len(gt),
            "correct_edges": len(correct),
            "false_edges": len(false),
            "score": len(correct) - len(false),
            "precision": prec,
            "recall": rec
        }

    # ---- Plan parsing/queue (optional; only if drones themselves keep PLAN in memory)
    def _maybe_update_plan_from_text(self, drone_id: int, text: str):
        if update_plan_from_text(self.plans, drone_id, text or ""):
            self.post_info(f"[Plan] Drone {drone_id} plan set: {self.plans[drone_id]}")
        else:
            if "PLAN" in (text or "").upper() or "PATH=" in (text or ""):
                LOGGER.log(f"Plan text found but unparsed for Drone {drone_id}.")

    def _next_planned_step(self, drone_id: int) -> Optional[str]:
        q = self.plans.get(drone_id, [])
        return q[0] if q else None

    def _advance_plan(self, drone_id: int):
        q = self.plans.get(drone_id, [])
        if q:
            q.pop(0)

    # ---- Edge logging
    def _log_edge_line(self, edge: Tuple[Tuple[int,int], Tuple[int,int]]):
        src, dst = edge
        is_correct = edge in self.gt_edges
        tag = "CORRECT" if is_correct else "FALSE"
        msg = f"Discovered edge: {src} -> {dst} [{tag}]"
        LOGGER.log(msg)
        self.post_info(msg)

    def _log_discovered_edges_incremental(self):
        current = self.discovered_edges()
        new_edges = current - self._edge_log_seen
        if not new_edges:
            return
        for e in sorted(new_edges):
            self._log_edge_line(e)
        self._edge_log_seen |= new_edges

    def _log_final_summary(self):
        disc = self.discovered_edges()
        correct = disc & self.gt_edges
        false = disc - self.gt_edges
        prec = (len(correct) / len(disc)) if disc else 0.0
        rec = (len(correct) / len(self.gt_edges)) if self.gt_edges else 0.0

        LOGGER.log("#" * 60)
        LOGGER.log("FINAL EDGE SUMMARY")
        LOGGER.log(f"Identified nodes: {len({n for e in disc for n in e})}")
        LOGGER.log(f"GT edges:         {len(self.gt_edges)}")
        LOGGER.log(f"Discovered edges: {len(disc)}")
        LOGGER.log(f"  - Correct:      {len(correct)}")
        LOGGER.log(f"  - False:        {len(false)}")
        LOGGER.log(f"Score (correct - false): {len(correct) - len(false)}")
        LOGGER.log(f"Precision: {prec:.3f}  |  Recall: {rec:.3f}")
        if false:
            LOGGER.log(f"False edges list ({len(false)}):")
            for e in sorted(false):
                LOGGER.log(f"  {e[0]} -> {e[1]}")

        self.post_info("=== FINAL EDGE SUMMARY ===")
        self.post_info(f"Disc:{len(disc)}  Corr:{len(correct)}  False:{len(false)}  "
                       f"Score:{len(correct)-len(false)}  P:{prec:.2f} R:{rec:.2f}")

    def _auto_replan_if_illegal(self, drone_id: int):
        """If next planned step is illegal/OOB, drop it and insert a legal detour that increases coverage."""
        d = next((dr for dr in self.drones if dr.id == drone_id), None)
        if not d: return
        q = self.plans.get(drone_id, [])
        if not q: return
        nxt = q[0]
        if nxt not in d._allowed_directions():
            # drop head
            q.pop(0)
            # pick a legal move that moves away from current center and avoids re-co-location
            allowed = d._allowed_directions()
            if not allowed: return
            x,y = d.position
            tile = self.board[x][y]
            occupied_dirs = set()
            for name,(dx,dy) in DIRECTION_MAP.items():
                nx,ny = x+dx,y+dy
                if 0 <= nx < CONFIG["board"]["width"] and 0 <= ny < CONFIG["board"]["height"]:
                    if self.board[nx][ny].drones and name in allowed:
                        occupied_dirs.add(name)
            candidates = [a for a in sorted(allowed) if a not in occupied_dirs]
            detour = candidates[0] if candidates else sorted(allowed)[0]
            q.insert(0, detour)
            self.plans[drone_id] = q
            self.post_info(f"[Plan] Drone {drone_id} auto-replan: inserted detour '{detour}'")

    def _is_valid_broadcast_json(self, msg: str) -> bool:
        try:
            obj = json.loads(msg)
            if not isinstance(obj, dict): return False
            if "obs" in obj:
                o = obj["obs"]
                return isinstance(o, dict) and "x" in o and "y" in o and "here" in o and "neighbors" in o
            if "plan" in obj:
                p = obj["plan"]
                return isinstance(p, dict) and "queue" in p
            return False
        except Exception:
            return False

    # ---- GUI/log helper
    def post_info(self, msg: str):
        if hasattr(self, "gui"):
            self.gui.post_info(msg)

    def current_drone(self) -> Optional[_Drone]:
        return self.drones[self.turn - 1] if 1 <= self.turn <= len(self.drones) else None

    # ---- Turn orchestration (no coaching, no global nudges, no auto-retargets)
    def _start_drone_turn(self, drone: _Drone):
        self._thinking = True
        self._current_future = self.executor.submit(drone.generate_full_model_response)

    def _try_finish_drone_turn(self, drone: _Drone) -> bool:
        if self._current_future is None or not self._current_future.done():
            return False

        try:
            messages = self._current_future.result()
            result = messages[-1]["content"]
            LOGGER.log(f"Drone {drone.id} response:\n{pprint.pformat(result, indent=4, width=200)}")

            # update plans from memory/message
            self._maybe_update_plan_from_text(drone.id, result.get("memory",""))
            self._maybe_update_plan_from_text(drone.id, result.get("message",""))
            self._auto_replan_if_illegal(drone.id)

            # ingest edges
            raw_edges = _normalize_edges(result.get("found_edges"))
            fedges: Set[Tuple[Tuple[int,int], Tuple[int,int]]] = set()
            if raw_edges:
                # keep only edges that are locally plausible for THIS drone at THIS turn
                src_expected = drone.position
                for (src, dst) in raw_edges:
                    if src == src_expected and _is_edge_locally_plausible(self.board, src, dst):
                        fedges.add((src, dst))
                    else:
                        self.post_info(f"Discarded implausible edge from Drone {drone.id}: {src} -> {dst}")

            if fedges:
                cur = self.drone_edges.get(drone.id, set())
                before = len(cur)
                cur |= fedges
                self.drone_edges[drone.id] = cur
                added = len(cur) - before
                if added > 0:
                    self.post_info(f"Drone {drone.id} submitted {added} plausible edge(s).")

            self.post_info(f"Drone {drone.id}:")
            self.post_info(f"Rationale: {result.get('rationale','')}")
            action = result.get("action", "wait")
            phase = self.phase_label()

            # DS prefer-top gating
            ds_cfg = CONFIG.get("decision_support", {})
            if bool(ds_cfg.get("prefer_top_recommendation", True)) and action == "move":
                top = self._ds_top_by_drone.get(drone.id)
                result_dir = (result.get("direction") or "").lower()
                expected = self._next_planned_step(drone.id)
                plan_viol = bool(expected) and (result_dir != expected)
                not_top = (not top) or (top.get("action") != "move") or ((top.get("specifier") or "") != result_dir)
                if plan_viol and not_top:
                    self.post_info("DecisionSupport gating: non-top, plan-violating action → waiting.")
                    action = "wait"

            # planning blocks movement
            if phase == "Planning" and action == "move":
                self.post_info("Planning phase: movement disabled. Waiting.")
                action = "wait"

            if action == "move":
                direction = (result.get("direction") or "").lower()
                allowed = drone._allowed_directions()
                if direction not in allowed:
                    self.post_info(f"Invalid/OOB direction '{direction}' (allowed={allowed}). Waiting.")
                else:
                    if phase == "Execution" and self.enforce_plan:
                        expected = self._next_planned_step(drone.id)
                        if expected and direction != expected:
                            self.post_info(f"Deviation from plan: expected '{expected}', got '{direction}'. Waiting.")
                        else:
                            if drone._move(direction):
                                self.post_info(f"Move {direction} to {drone.position}")
                                if expected == direction:
                                    self._advance_plan(drone.id)
                    else:
                        if drone._move(direction):
                            self.post_info(f"Move {direction} to {drone.position}")

            elif action == "broadcast":
                msg = (result.get("message") or "").strip()
                if not msg or not self._is_valid_broadcast_json(msg):
                    self.post_info("Invalid broadcast (empty or non-JSON). Waiting.")
                else:
                    self.post_info("Broadcast")
                    self.post_info(msg)
                    tile = self.board[drone.position[0]][drone.position[1]]
                    for d in tile.drones:
                        if d.id != drone.id:
                            d.rx_buffer += f"Drone {drone.id} broadcasted: {msg}\n"

            else:
                self.post_info("Wait")

            # persist memory and mark visited
            mem_txt = (result.get("memory") or "").strip()
            if mem_txt:
                drone.memory = mem_txt
            vx, vy = drone.position
            token = f"VISITED:{vx},{vy}"
            if token not in drone.memory:
                drone.memory += ("" if drone.memory.endswith("\n") else "\n") + token

            self._log_discovered_edges_incremental()
            self.post_info("\n")
        except Exception as e:
            LOGGER.log(f"Error finishing Drone {drone.id}'s turn: {e}")

        self._thinking = False
        self._current_future = None
        return True

    # ---- Main loop
    def run_simulation(self):
        max_rounds = CONFIG["simulation"].get("max_rounds", 10)
        use_gui = CONFIG["simulation"].get("use_gui", True)

        running = True
        clock = pygame.time.Clock()

        if use_gui and hasattr(self, "gui"):
            pygame.display.set_caption(f"Simulation - Round {1}.1/{max_rounds}.{self.num_drones}")
            self.gui.draw_field()

        current_round = 1
        drone_index = 0
        pending = False

        try:
            while running:
                if use_gui and hasattr(self, "gui"):
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT: running = False
                        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False

                if not pending:
                    if current_round > max_rounds:
                        break
                    self.round = current_round
                    self.turn = drone_index + 1
                    caption = f"Simulation - Round {current_round}.{self.turn}/{max_rounds}.{self.num_drones}"
                    LOGGER.log('#'*50); LOGGER.log(caption + f" | Phase: {self.phase_label()}")
                    if use_gui and hasattr(self, "gui"):
                        pygame.display.set_caption(caption)

                    self._start_drone_turn(self.drones[drone_index])
                    pending = True

                if pending:
                    d = self.drones[drone_index]
                    if self._try_finish_drone_turn(d):
                        drone_index += 1
                        if drone_index >= self.num_drones:
                            drone_index = 0
                            current_round += 1
                        pending = False

                if use_gui and hasattr(self, "gui"):
                    self.gui.draw_field()

                clock.tick(60)

        except KeyboardInterrupt:
            LOGGER.log("KeyboardInterrupt received — shutting down gracefully.")
            running = False
        finally:
            try:
                self._log_final_summary()
            except Exception:
                pass
            self.shutdown()

    def shutdown(self):
        try:
            if getattr(self, "_current_future", None) and not self._current_future.done():
                self._current_future.cancel()
        except Exception:
            pass
        try:
            if getattr(self, "executor", None):
                self.executor.shutdown(wait=False, cancel_futures=True)
                self.executor = None
        except Exception:
            pass
        try:
            if hasattr(self, "gui"):
                pygame.display.quit()
            pygame.quit()
        except Exception:
            pass
        LOGGER.log("Clean shutdown complete.")


# In[30]:


# Main

if __name__ == "__main__":
    try:
        LOGGER.log("Launching simulation.")
        SIM = Simulation()
        SIM.run_simulation()
    except KeyboardInterrupt:
        LOGGER.log("Interrupted by user (Ctrl+C).")
        try: SIM.shutdown()
        except Exception: pass

