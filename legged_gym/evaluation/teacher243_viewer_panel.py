"""Small sidecar control panel for the Isaac Gym Teacher243 viewer."""

import queue
import tkinter as tk
from tkinter import ttk

from .teacher243_viewer_overlay import (
    ACTUAL_PATH_COLOR,
    SEVERITY_STYLES,
    TARGET_PATH_COLOR,
)


def _rgb_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*(int(round(value * 255)) for value in rgb))


def run_panel(control_queue, status_queue, num_envs):
    root = tk.Tk()
    root.title("Teacher243 Failure Viewer")
    root.update_idletasks()
    panel_width = 430
    panel_x = max(0, root.winfo_screenwidth() - panel_width - 24)
    root.geometry("{}x650+{}+70".format(panel_width, panel_x))
    root.minsize(410, 560)

    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure("Header.TLabel", font=("TkDefaultFont", 12, "bold"))
    style.configure("Section.TLabel", font=("TkDefaultFont", 10, "bold"))

    state = {
        "show_target": tk.BooleanVar(value=True),
        "show_actual": tk.BooleanVar(value=True),
        "show_degradation": tk.BooleanVar(value=True),
        "selected_only": tk.BooleanVar(value=False),
        "selected_env": tk.IntVar(value=0),
    }

    def send_state(extra=None):
        payload = {
            "show_target": state["show_target"].get(),
            "show_actual": state["show_actual"].get(),
            "show_degradation": state["show_degradation"].get(),
            "selected_only": state["selected_only"].get(),
            "selected_env": state["selected_env"].get(),
        }
        if extra:
            payload.update(extra)
        try:
            control_queue.put_nowait(payload)
        except queue.Full:
            pass

    outer = ttk.Frame(root, padding=12)
    outer.pack(fill="both", expand=True)
    ttk.Label(outer, text="Teacher243 Failure Viewer", style="Header.TLabel").pack(anchor="w")
    ttk.Label(
        outer,
        text="Green: commanded-motion reference   Blue: robot trajectory",
    ).pack(anchor="w", pady=(2, 10))

    controls = ttk.LabelFrame(outer, text="Display", padding=8)
    controls.pack(fill="x")
    for key, label in (
        ("show_target", "Target trajectory"),
        ("show_actual", "Actual trajectory"),
        ("show_degradation", "Degraded-link color"),
        ("selected_only", "Selected robot paths only"),
    ):
        ttk.Checkbutton(
            controls, text=label, variable=state[key], command=send_state
        ).pack(anchor="w")
    ttk.Button(
        controls, text="Clear trajectories", command=lambda: send_state({"clear": True})
    ).pack(fill="x", pady=(7, 0))
    ttk.Button(
        controls,
        text="Focus selected robot",
        command=lambda: send_state({"focus": True}),
    ).pack(fill="x", pady=(4, 0))

    ttk.Label(outer, text="Robots", style="Section.TLabel").pack(anchor="w", pady=(12, 4))
    tree = ttk.Treeview(
        outer,
        columns=("env", "joint", "d", "output", "effective"),
        show="headings",
        height=min(max(int(num_envs), 3), 8),
        selectmode="browse",
    )
    columns = (
        ("env", "Robot", 55),
        ("joint", "Degraded joint", 160),
        ("d", "d", 55),
        ("output", "1-d", 55),
        ("effective", "Effective", 70),
    )
    for name, title, width in columns:
        tree.heading(name, text=title)
        tree.column(name, width=width, anchor="center", stretch=name == "joint")
    tree.pack(fill="x")

    updating_rows = False

    def on_select(_event=None):
        if updating_rows:
            return
        selection = tree.selection()
        if selection:
            state["selected_env"].set(int(tree.item(selection[0], "values")[0]))
            # Selecting a row changes only the displayed path selection.
            # Camera motion is deliberately reserved for the explicit Focus
            # button so respawn/status refreshes never steal camera control.
            send_state()

    tree.bind("<<TreeviewSelect>>", on_select)

    ttk.Label(outer, text="Degradation legend", style="Section.TLabel").pack(
        anchor="w", pady=(12, 4)
    )
    legend = ttk.Frame(outer)
    legend.pack(fill="x")
    for style_item in SEVERITY_STYLES:
        row = ttk.Frame(legend)
        row.pack(fill="x", pady=1)
        swatch = tk.Label(
            row,
            width=3,
            height=1,
            bg=style_item.hex_color,
            relief="solid",
            borderwidth=1,
        )
        swatch.pack(side="left", padx=(0, 7))
        ttk.Label(
            row,
            text="d={:.1f}   {:>3.0f}% output   {}".format(
                style_item.degradation,
                (1.0 - style_item.degradation) * 100.0,
                style_item.name,
            ),
        ).pack(side="left")

    path_legend = ttk.Frame(outer)
    path_legend.pack(fill="x", pady=(8, 0))
    for color, label in (
        (_rgb_hex(TARGET_PATH_COLOR), "Commanded-motion reference"),
        (_rgb_hex(ACTUAL_PATH_COLOR), "Actual trajectory"),
    ):
        row = ttk.Frame(path_legend)
        row.pack(fill="x", pady=1)
        tk.Label(row, width=3, height=1, bg=color, relief="solid", borderwidth=1).pack(
            side="left", padx=(0, 7)
        )
        ttk.Label(row, text=label).pack(side="left")

    status_text = tk.StringVar(value="Waiting for simulator...")
    ttk.Label(outer, textvariable=status_text).pack(anchor="w", pady=(10, 0))

    row_tags = set()

    def update_rows(rows):
        nonlocal updating_rows
        nonlocal row_tags
        updating_rows = True
        for item in tree.get_children():
            tree.delete(item)
        row_tags = set()
        for row in rows:
            tag = "severity_{}".format(row["env"])
            row_tags.add(tag)
            foreground = "#FFFFFF" if row["degradation"] >= 0.8 else "#111111"
            tree.tag_configure(
                tag, background=row["color"], foreground=foreground
            )
            tree.insert(
                "",
                "end",
                iid="env_{}".format(row["env"]),
                tags=(tag,),
                values=(
                    row["env"],
                    row["joint"],
                    "{:.1f}".format(row["degradation"]),
                    "{:.0f}%".format(row["remaining"] * 100.0),
                    "{:.0f}%".format(row["effective_strength"] * 100.0),
                ),
            )
        selected = "env_{}".format(state["selected_env"].get())
        if tree.exists(selected):
            tree.selection_set(selected)
        updating_rows = False

    def poll_status():
        latest = None
        try:
            while True:
                latest = status_queue.get_nowait()
        except queue.Empty:
            pass
        if latest:
            if "rows" in latest:
                update_rows(latest["rows"])
            if "message" in latest:
                status_text.set(latest["message"])
        root.after(150, poll_status)

    def close():
        send_state({"quit": True})
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    send_state()
    poll_status()
    root.mainloop()
