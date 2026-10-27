#!/usr/bin/env python3
"""
Robot Pipeline App
Onshape -> URDF -> cleaned URDF -> Isaac Lab USD.

Run:  python3 robot_pipeline_app.py
Requires: Python 3 with tkinter, onshape-to-robot, a conda env with Isaac Lab, and a
folder containing tools/rebuild_urdf.py (your standing_policy folder).
"""
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import tkinter as tk
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

SETTINGS_FILE = os.path.expanduser("~/.robot_pipeline_settings.json")
# The secret key is never written to disk.
SAVED_FIELDS = [
    "robot_name", "onshape_url", "access_key", "api_url", "project_dir",
    "isaaclab_dir", "conda_env",
]

# Actuator values written into every joint (GIM8108-8). Keep equal to TORQUE_LIMIT_NM /
# NO_LOAD_SPEED_RAD_S in biped_stand/robot_cfg.py.
EFFORT_NM = 20.0
VELOCITY_RAD_S = 16.0


def rebuild_urdf(src, out, log, limits_file=None):
    """Turn a raw onshape-to-robot export into an Isaac-ready URDF (ASCII names, real
    limits, real torque). Reads <src>/robot.urdf + <src>/assets/*.stl (never modified);
    writes <out>/robot.urdf + <out>/assets/m_*.stl (the out folder is wiped and recreated).
    Raises RuntimeError on a hard problem. Returns the number of warnings."""

    def clean(s):
        s = re.sub(r"[^A-Za-z0-9_]", "_", s)
        return re.sub(r"_+", "_", s).strip("_")

    def norm(s):
        return unicodedata.normalize("NFC", s)

    src, out = Path(src).expanduser().resolve(), Path(out).expanduser().resolve()
    if out == src or (src in out.parents and out.name in ("", "assets")):
        raise RuntimeError("Output folder must be separate from the source (it is wiped).")
    tree = ET.parse(src / "robot.urdf")
    root = tree.getroot()

    # ---- meshes ----
    on_disk = {norm(p.name): p for p in (src / "assets").glob("*.stl")}
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)
    done = {}
    for mesh in root.iter("mesh"):
        fn = norm(mesh.attrib["filename"])
        base = fn.split("/")[-1]
        if base not in on_disk:
            raise RuntimeError(f"mesh {base!r} referenced by the URDF is not in {src / 'assets'}")
        if base not in done:
            new = "m_" + clean(Path(base).stem) + ".stl"
            while new in done.values():
                new = new[:-4] + "_x.stl"
            shutil.copy(on_disk[base], out / "assets" / new)
            done[base] = new
        mesh.attrib["filename"] = "assets/" + done[base]

    # ---- materials ----
    for m in root.iter("material"):
        if "name" in m.attrib:
            m.attrib["name"] = "mat_" + clean(m.attrib["name"])

    # ---- joints ----
    overrides = json.loads(Path(limits_file).read_text()) if limits_file else {}
    log(f"{'joint':<20}{'lower':>9}{'upper':>9}  note\n")
    problems = 0
    for j in root.findall("joint"):
        name = j.attrib["name"]
        if j.attrib["type"] == "fixed":
            continue
        lim = j.find("limit")
        if lim is None:
            lim = ET.SubElement(j, "limit")
        if name in overrides:
            lim.attrib["lower"], lim.attrib["upper"] = repr(float(overrides[name][0])), repr(float(overrides[name][1]))
        lo, hi = float(lim.attrib.get("lower", -math.pi)), float(lim.attrib.get("upper", math.pi))
        lim.attrib["effort"], lim.attrib["velocity"] = str(EFFORT_NM), str(VELOCITY_RAD_S)
        note = ""
        if hi - lo > 2 * math.pi - 0.05:
            note = "PLACEHOLDER (+-pi): set real limits in the Onshape mate or joint_limits.json"
            problems += 1
        elif name in overrides:
            note = "from joint_limits.json"
        log(f"{name:<20}{lo:>9.3f}{hi:>9.3f}  {note}\n")

    # ---- sanity ----
    total = 0.0
    for l in root.findall("link"):
        i = l.find("inertial")
        m = float(i.find("mass").attrib["value"]) if i is not None else 0.0
        total += m
        if i is None or m <= 0:
            log(f"WARNING: link {l.attrib['name']} has no/zero mass\n")
            problems += 1
    log(f"total mass {total:.3f} kg (battery, Pi and wiring are not in the CAD unless modelled)\n")

    ET.indent(tree, space="  ")
    tree.write(out / "robot.urdf", encoding="utf-8", xml_declaration=True)
    bad = [n for n, line in enumerate((out / "robot.urdf").read_text(encoding="utf-8").splitlines(), 1)
           if any(ord(c) > 127 for c in line)]
    if bad:
        log(f"WARNING: non-ASCII characters remain on lines {bad}\n")
        problems += 1
    log(f"wrote {out / 'robot.urdf'} and {len(done)} meshes; {problems} warning(s)\n")
    return problems


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Robot Pipeline: Onshape to USD")
        self.geometry("900x700")
        self.proc = None
        self.logfile = None
        self.stop_flag = False

        self.vars = {
            "robot_name": tk.StringVar(value="biped"),
            "onshape_url": tk.StringVar(),
            "access_key": tk.StringVar(),
            "secret_key": tk.StringVar(),
            "api_url": tk.StringVar(value="https://cad.onshape.com"),
            "project_dir": tk.StringVar(),
            "isaaclab_dir": tk.StringVar(value="~/IsaacLab"),
            "conda_env": tk.StringVar(value="env_isaaclab"),
        }
        self.stages = {
            "urdf": tk.BooleanVar(value=True),
            "clean": tk.BooleanVar(value=True),
            "usd": tk.BooleanVar(value=True),
            "check": tk.BooleanVar(value=True),
        }
        self.load_settings()
        self.build_ui()

    # ---------- UI ----------
    def build_ui(self):
        pad = {"padx": 8, "pady": 3}
        form = ttk.LabelFrame(self, text="Inputs")
        form.pack(fill="x", padx=10, pady=8)
        form.columnconfigure(1, weight=1)

        def row(r, label, key, show=None, browse=False):
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w", **pad)
            ttk.Entry(form, textvariable=self.vars[key], show=show).grid(
                row=r, column=1, sticky="ew", **pad)
            if browse:
                ttk.Button(form, text="Browse...", command=lambda: self.browse(key)).grid(
                    row=r, column=2, **pad)

        row(0, "Robot name", "robot_name")
        row(1, "Onshape link", "onshape_url")
        row(2, "Onshape access key", "access_key")
        row(3, "Onshape secret key", "secret_key", show="*")
        row(4, "Project folder path", "project_dir", browse=True)

        adv = ttk.LabelFrame(self, text="Advanced (defaults match your notes)")
        adv.pack(fill="x", padx=10, pady=4)
        adv.columnconfigure(1, weight=1)
        for r, (label, key, br) in enumerate([
            ("Onshape API URL", "api_url", False),
            ("Isaac Lab folder", "isaaclab_dir", True),
            ("Conda env", "conda_env", False),
        ]):
            ttk.Label(adv, text=label).grid(row=r, column=0, sticky="w", **pad)
            ttk.Entry(adv, textvariable=self.vars[key]).grid(row=r, column=1, sticky="ew", **pad)
            if br:
                ttk.Button(adv, text="Browse...", command=lambda k=key: self.browse(k)).grid(
                    row=r, column=2, **pad)

        st = ttk.LabelFrame(self, text="Stages to run")
        st.pack(fill="x", padx=10, pady=4)
        labels = [
            ("urdf", "1. Onshape -> URDF"),
            ("clean", "2. Clean URDF"),
            ("usd", "3. Create USD"),
            ("check", "4. Check USD"),
        ]
        for i, (k, t) in enumerate(labels):
            ttk.Checkbutton(st, text=t, variable=self.stages[k]).grid(
                row=0, column=i, sticky="w", padx=10, pady=2)

        bf = ttk.Frame(self)
        bf.pack(fill="x", padx=10, pady=6)
        self.run_btn = ttk.Button(bf, text="Run pipeline", command=self.start)
        self.run_btn.pack(side="left")
        self.stop_btn = ttk.Button(bf, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        ttk.Button(bf, text="Clear log", command=lambda: self.log.delete("1.0", "end")).pack(side="left")
        ttk.Button(bf, text="Copy log", command=self.copy_log).pack(side="left", padx=8)

        self.log = tk.Text(self, bg="#111", fg="#ddd", insertbackground="#ddd",
                           font=("Menlo", 10), wrap="word")
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def browse(self, key):
        d = filedialog.askdirectory(initialdir=os.path.expanduser(self.vars[key].get() or "~"))
        if d:
            self.vars[key].set(d)

    # ---------- settings ----------
    def load_settings(self):
        try:
            with open(SETTINGS_FILE) as f:
                data = json.load(f)
            for k in SAVED_FIELDS:
                if k in data:
                    self.vars[k].set(data[k])
        except Exception:
            pass

    def save_settings(self):
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump({k: self.vars[k].get() for k in SAVED_FIELDS}, f, indent=2)
        except Exception:
            pass

    # ---------- logging ----------
    def write(self, text):
        if self.logfile:
            try:
                self.logfile.write(text)
                self.logfile.flush()
            except Exception:
                pass
        self.after(0, self._write, text)

    def _write(self, text):
        self.log.insert("end", text)
        self.log.see("end")

    def copy_log(self):
        self.clipboard_clear()
        self.clipboard_append(self.log.get("1.0", "end"))
        self.update()
        messagebox.showinfo("Copy log", "Log copied to clipboard.")

    # ---------- run ----------
    def validate(self):
        v = {k: x.get().strip() for k, x in self.vars.items()}
        errs = []
        if not v["robot_name"]:
            errs.append("Robot name is required.")
        if self.stages["urdf"].get():
            if not v["onshape_url"].startswith("http"):
                errs.append("Onshape link must be a full URL.")
            if not v["access_key"] or not v["secret_key"]:
                errs.append("Onshape access and secret keys are required.")
        if not v["project_dir"] or not os.path.isdir(os.path.expanduser(v["project_dir"])):
            errs.append("Project folder path must be an existing folder.")
        elif self.stages["clean"].get() and not os.path.isfile(
                os.path.join(os.path.expanduser(v["project_dir"]), "robot.urdf")) and not self.stages["urdf"].get():
            errs.append("No robot.urdf in the project folder. Enable stage 1 or run it first.")
        if errs:
            messagebox.showerror("Check inputs", "\n".join(errs))
            return None
        return v

    def start(self):
        v = self.validate()
        if not v:
            return
        self.save_settings()
        self.stop_flag = False
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        threading.Thread(target=self.pipeline, args=(v,), daemon=True).start()

    def stop(self):
        self.stop_flag = True
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.write("\n[Stopping...]\n")

    def sh(self, cmd, cwd=None, env=None):
        """Run a bash command, streaming output. Returns exit code."""
        self.write(f"\n$ {cmd}\n")
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        self.proc = subprocess.Popen(
            ["bash", "-lc", cmd], cwd=cwd, env=full_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in self.proc.stdout:
            self.write(line)
        return self.proc.wait()

    def pipeline(self, v):
        try:
            proj = os.path.abspath(os.path.expanduser(v["project_dir"]))
            lab = os.path.abspath(os.path.expanduser(v["isaaclab_dir"]))
            name = v["robot_name"]
            q = shlex.quote
            urdf = os.path.join(proj, "isaac_export", "robot.urdf")
            usd_dir = os.path.join(proj, "usd")
            usd = os.path.join(usd_dir, f"{name}.usd")
            conda = (f'source "$(conda info --base)/etc/profile.d/conda.sh" && '
                     f'conda deactivate 2>/dev/null; conda activate {q(v["conda_env"])}')
            s = self.stages
            self.logfile = open(os.path.join(proj, "pipeline.log"), "w", encoding="utf-8")
            self.write(f"Log also saved to {os.path.join(proj, 'pipeline.log')}\n")

            if s["urdf"].get():
                self.write("\n=== 1. Onshape -> URDF ===\n")
                cfg = {
                    "url": v["onshape_url"],
                    "output_format": "urdf",
                    "robot_name": name,
                    "output_filename": "robot",
                    "assets_directory": "assets",
                }
                with open(os.path.join(proj, "config.json"), "w") as f:
                    json.dump(cfg, f, indent=4)
                self.write("Wrote config.json\n")
                env = {
                    "ONSHAPE_API": v["api_url"],
                    "ONSHAPE_ACCESS_KEY": v["access_key"],
                    "ONSHAPE_SECRET_KEY": v["secret_key"],
                }
                if self.run_stage(f'onshape-to-robot {q(proj)}', cwd=proj, env=env):
                    return

            if s["clean"].get():
                self.write("\n=== 2. Clean URDF ===\n")
                lim_path = os.path.join(proj, "joint_limits.json")
                try:
                    rebuild_urdf(proj, os.path.join(proj, "isaac_export"), self.write,
                                 lim_path if os.path.isfile(lim_path) else None)
                except Exception as e:
                    self.write(f"\n[Clean URDF failed: {e}. Pipeline halted.]\n")
                    return

            if s["usd"].get():
                self.write("\n=== 3. Create USD ===\n")
                if os.path.isdir(usd_dir):  # keep the previous USD
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    bk = os.path.join(proj, "old_exports", stamp)
                    os.makedirs(bk, exist_ok=True)
                    shutil.move(usd_dir, os.path.join(bk, "usd"))
                    self.write(f"Previous usd/ moved to {bk}/usd\n")
                os.makedirs(usd_dir, exist_ok=True)
                cmd = (f'{conda} && cd {q(lab)} && ./isaaclab.sh -p scripts/tools/convert_urdf.py '
                       f'{q(urdf)} {q(usd)} --joint-stiffness 0.0 --joint-damping 0.0 '
                       f'--joint-target-type none --headless')
                if self.run_stage(cmd):
                    return

            if s["check"].get():
                self.write("\n=== 4. Check USD ===\n")
                self.write("Count of FlatGrid//Render lines (expect 0 for a clean USD):\n")
                # grep -c exits 1 on zero matches, so don't treat that as failure.
                self.sh(f'strings {q(usd)} | grep -cE "FlatGrid|/Render"')
                if self.stop_flag:
                    return

            self.write(f"\n=== Pipeline complete. USD: {usd} ===\n")
        except Exception as e:
            self.write(f"\n[Error] {e}\n")
        finally:
            if self.logfile:
                try:
                    self.logfile.close()
                except Exception:
                    pass
                self.logfile = None
            self.after(0, lambda: (self.run_btn.config(state="normal"),
                                   self.stop_btn.config(state="disabled")))

    def run_stage(self, cmd, cwd=None, env=None):
        """Returns True if the pipeline should halt."""
        rc = self.sh(cmd, cwd=cwd, env=env)
        if self.stop_flag:
            self.write("[Stopped by user]\n")
            return True
        if rc != 0:
            self.write(f"\n[Stage failed with exit code {rc}. Pipeline halted.]\n")
            return True
        return False


if __name__ == "__main__":
    App().mainloop()