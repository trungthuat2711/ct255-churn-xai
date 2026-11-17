"""Lưu trữ kết quả thực nghiệm (file .txt / .json / .csv) và truy vết nguồn code của từng hình/bảng.

Tham khảo cách ghi file: https://www.w3schools.com/python/python_file_write.asp
  - open(path, "w") : ghi đè    - open(path, "a") : ghi nối tiếp
Mỗi hình/bảng được đăng ký vào `artifact_index.csv` cùng với:
  file code : hàm : dòng sinh ra nó  ->  dùng để gắn link code (GitHub) vào hình/bảng trong báo cáo.
"""
import csv
import datetime
import hashlib
import inspect
import json
import os
import platform
import sys


def sha256_of(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def to_jsonable(o):
    """Chuyển numpy/pandas/obj bất kỳ sang kiểu JSON an toàn."""
    import numpy as np
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def env_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "shap", "imblearn", "matplotlib", "scipy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = "not installed"
    return info


class ExperimentLogger:
    def __init__(self, root_dir, exp_id, command):
        self.root_dir = root_dir
        self.exp_id = exp_id
        self.command = command
        self.start = datetime.datetime.now()
        os.makedirs(root_dir, exist_ok=True)
        self.artifacts = []          # danh sách hình/bảng đã tạo

    # ---------- ghi file cơ bản ----------
    def write_text(self, path, text, mode="w"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, mode, encoding="utf-8") as f:
            f.write(text)

    def save_json(self, path, obj, desc="", model=""):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(to_jsonable(obj), indent=2, ensure_ascii=False))
        self.add_artifact(path, desc, model, depth=2)

    def save_csv(self, path, df, desc="", model="", index=False):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=index, encoding="utf-8")
        self.add_artifact(path, desc, model, depth=2)

    def save_fig(self, fig, path, desc="", model="", dpi=150):
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        self.add_artifact(path, desc, model, depth=2)

    # ---------- truy vết nguồn code ----------
    def add_artifact(self, path, desc="", model="", depth=1):
        fr = inspect.stack()[depth]
        src = f"{os.path.relpath(fr.filename, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))}:{fr.function}:L{fr.lineno}"
        self.artifacts.append({
            "artifact": os.path.relpath(path, self.root_dir),
            "type": os.path.splitext(path)[1].lstrip("."),
            "model": model,
            "description": desc,
            "generated_by": src,
            "command": self.command,
            "sha256": sha256_of(path) if os.path.exists(path) else "",
        })

    def write_artifact_index(self):
        path = os.path.join(self.root_dir, "artifact_index.csv")
        cols = ["artifact", "type", "model", "description", "generated_by", "command", "sha256"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for a in self.artifacts:
                if a["artifact"] != "artifact_index.csv":
                    a["sha256"] = sha256_of(os.path.join(self.root_dir, a["artifact"]))
                    w.writerow(a)
        return path

    def append_global_registry(self, output_parent, line):
        """Ghi nối tiếp (mode 'a') một dòng vào file tổng hợp mọi lần chạy."""
        os.makedirs(output_parent, exist_ok=True)
        with open(os.path.join(output_parent, "all_experiments.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ---------- định dạng log dạng văn bản dễ đọc ----------
def fmt_block(title, d, indent=0):
    pad = "  " * indent
    lines = [f"{pad}{title}"] if title else []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(fmt_block(f"{k}:", v, indent + 1))
        elif isinstance(v, float):
            lines.append(f"{pad}  {k}: {v:.6f}")
        else:
            lines.append(f"{pad}  {k}: {v}")
    return "\n".join(lines)


def banner(text, ch="="):
    line = ch * 78
    return f"\n{line}\n{text}\n{line}\n"
