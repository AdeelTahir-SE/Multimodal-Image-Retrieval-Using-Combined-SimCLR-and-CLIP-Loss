import os
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def _ensure_project_venv_python() -> None:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"

    if not venv_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    if current_python == venv_python.resolve():
        return

    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_ensure_project_venv_python()

import torch
from PIL import Image, ImageTk
from transformers import DistilBertTokenizer

from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from retrieve import (
    embed_image_query,
    embed_text_query,
    load_config,
    load_or_build_gallery_cache,
    topk_matches,
)


class RetrievalUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Visual Retrieval Studio")
        self.root.geometry("1380x900")
        self.root.minsize(1100, 720)
        self._configure_style()

        self.gallery_index = None
        self.gallery_paths = []
        self.cache_path = None
        self.image_encoder = None
        self.text_encoder = None
        self.tokenizer = None
        self.device = None
        self.cfg = None
        self.image_refs = []
        self.query_preview_ref = None
        self.cards_per_row = 4
        self.current_result_items = []

        self._build_layout()
        self.root.bind("<Return>", lambda _e: self.search())

    def _configure_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        bg = "#f4f6f8"
        panel_bg = "#ffffff"
        accent = "#0b6e99"
        text = "#243447"

        self.root.configure(bg=bg)
        style.configure("Root.TFrame", background=bg)
        style.configure("Panel.TFrame", background=panel_bg)
        style.configure("Card.TFrame", background=panel_bg, relief="ridge", borderwidth=1)
        style.configure("Title.TLabel", background=bg, foreground="#0f1720", font=("Segoe UI Semibold", 18))
        style.configure("SubTitle.TLabel", background=bg, foreground="#486078", font=("Segoe UI", 10))
        style.configure("PanelLabel.TLabel", background=panel_bg, foreground=text, font=("Segoe UI", 10))
        style.configure("Status.TLabel", background="#e9f5fb", foreground="#12465f", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 10), padding=(10, 7))
        style.map("Accent.TButton", background=[("active", "#0f88bd"), ("!disabled", accent)], foreground=[("!disabled", "white")])
        style.configure("TLabelframe", background=panel_bg)
        style.configure("TLabelframe.Label", background=panel_bg, foreground="#2b3e50", font=("Segoe UI Semibold", 10))

    def _start_busy(self, message: str):
        self._set_busy(True)
        self.status_var.set(message)
        self.progress.start(12)

    def _stop_busy(self, message: str | None = None):
        self.progress.stop()
        self._set_busy(False)
        if message is not None:
            self.status_var.set(message)

    def _run_background(self, worker, busy_message: str):
        self._start_busy(busy_message)
        threading.Thread(target=worker, daemon=True).start()

    def _build_layout(self):
        root_wrap = ttk.Frame(self.root, style="Root.TFrame", padding=12)
        root_wrap.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root_wrap, style="Root.TFrame")
        header.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(header, text="Visual Retrieval Studio", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Search with text or image and preview ranked matches instantly",
            style="SubTitle.TLabel",
        ).pack(anchor="w")

        content = ttk.Frame(root_wrap, style="Root.TFrame")
        content.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(content, style="Panel.TFrame", padding=10)
        left.pack(side="left", fill=tk.Y, padx=(0, 10))

        right = ttk.Frame(content, style="Panel.TFrame", padding=10)
        right.pack(side="left", fill=tk.BOTH, expand=True)

        source_frame = ttk.LabelFrame(left, text="Sources", padding=8)
        source_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(source_frame, text="Config", style="PanelLabel.TLabel").grid(row=0, column=0, sticky="w")
        self.config_var = tk.StringVar(value="collab.config.yaml")
        ttk.Entry(source_frame, textvariable=self.config_var, width=44).grid(row=1, column=0, sticky="we", pady=(0, 6))

        ttk.Label(source_frame, text="Checkpoint", style="PanelLabel.TLabel").grid(row=2, column=0, sticky="w")
        self.checkpoint_var = tk.StringVar(value="checkpoints/best.pt")
        ttk.Entry(source_frame, textvariable=self.checkpoint_var, width=44).grid(row=3, column=0, sticky="we", pady=(0, 6))

        ttk.Label(source_frame, text="Image Directory", style="PanelLabel.TLabel").grid(row=4, column=0, sticky="w")
        self.image_dir_var = tk.StringVar(value="data/coco/val2017")
        ttk.Entry(source_frame, textvariable=self.image_dir_var, width=44).grid(row=5, column=0, sticky="we", pady=(0, 6))

        ttk.Label(source_frame, text="Cache Directory", style="PanelLabel.TLabel").grid(row=6, column=0, sticky="w")
        self.cache_dir_var = tk.StringVar(value=".cache/retrieval")
        ttk.Entry(source_frame, textvariable=self.cache_dir_var, width=44).grid(row=7, column=0, sticky="we", pady=(0, 6))

        config_row = ttk.Frame(source_frame, style="Panel.TFrame")
        config_row.grid(row=8, column=0, sticky="we", pady=(2, 4))
        ttk.Label(config_row, text="Batch", style="PanelLabel.TLabel").pack(side="left")
        self.batch_var = tk.StringVar(value="64")
        ttk.Entry(config_row, textvariable=self.batch_var, width=8).pack(side="left", padx=(6, 12))
        ttk.Label(config_row, text="Top K", style="PanelLabel.TLabel").pack(side="left")
        self.topk_var = tk.StringVar(value="6")
        ttk.Entry(config_row, textvariable=self.topk_var, width=8).pack(side="left", padx=(6, 0))

        self.rebuild_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(source_frame, text="Rebuild cache", variable=self.rebuild_var).grid(row=9, column=0, sticky="w")

        self.load_btn = ttk.Button(source_frame, text="Load Index", style="Accent.TButton", command=self.load_index)
        self.load_btn.grid(row=10, column=0, sticky="we", pady=(8, 2))
        source_frame.columnconfigure(0, weight=1)

        query_frame = ttk.LabelFrame(left, text="Query", padding=8)
        query_frame.pack(fill=tk.X, pady=(0, 8))

        self.query_mode = tk.StringVar(value="text")
        mode_row = ttk.Frame(query_frame, style="Panel.TFrame")
        mode_row.grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Radiobutton(
            mode_row,
            text="Text",
            variable=self.query_mode,
            value="text",
            command=self._update_query_mode_state,
        ).pack(side="left")
        ttk.Radiobutton(
            mode_row,
            text="Image Path",
            variable=self.query_mode,
            value="image",
            command=self._update_query_mode_state,
        ).pack(side="left", padx=(10, 0))

        ttk.Label(query_frame, text="Text Query", style="PanelLabel.TLabel").grid(row=1, column=0, sticky="w")
        self.text_query_var = tk.StringVar()
        self.text_query_entry = ttk.Entry(query_frame, textvariable=self.text_query_var, width=44)
        self.text_query_entry.grid(row=2, column=0, sticky="we", pady=(0, 6))

        ttk.Label(query_frame, text="Image Query Path", style="PanelLabel.TLabel").grid(row=3, column=0, sticky="w")
        image_path_row = ttk.Frame(query_frame, style="Panel.TFrame")
        image_path_row.grid(row=4, column=0, sticky="we", pady=(0, 6))
        self.image_query_var = tk.StringVar(value="data/coco/val2017/000000033368.jpg")
        self.image_query_entry = ttk.Entry(image_path_row, textvariable=self.image_query_var, width=34)
        self.image_query_entry.pack(side="left", fill=tk.X, expand=True)
        ttk.Button(image_path_row, text="Browse", command=self.browse_query_image).pack(side="left", padx=(6, 0))

        self.preview_canvas = tk.Canvas(query_frame, width=250, height=180, bg="#dde7ee", highlightthickness=0)
        self.preview_canvas.grid(row=5, column=0, sticky="we", pady=(2, 6))
        self.preview_canvas.create_text(125, 90, text="Query image preview", fill="#4b6578", font=("Segoe UI", 10))

        query_action_row = ttk.Frame(query_frame, style="Panel.TFrame")
        query_action_row.grid(row=6, column=0, sticky="we")
        self.search_btn = ttk.Button(query_action_row, text="Search", style="Accent.TButton", command=self.search)
        self.search_btn.pack(side="left", fill=tk.X, expand=True)
        ttk.Button(query_action_row, text="Clear", command=self.clear_results).pack(side="left", padx=(6, 0))
        query_frame.columnconfigure(0, weight=1)

        self.info_var = tk.StringVar(value="Index not loaded")
        ttk.Label(left, textvariable=self.info_var, style="PanelLabel.TLabel", wraplength=320, justify="left").pack(
            fill=tk.X,
            pady=(0, 6),
        )

        status_row = ttk.Frame(right, style="Panel.TFrame")
        status_row.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_row, textvariable=self.status_var, style="Status.TLabel", padding=(8, 6)).pack(
            side="left", fill=tk.X, expand=True
        )
        self.progress = ttk.Progressbar(status_row, mode="indeterminate", length=180)
        self.progress.pack(side="left", padx=(8, 0))

        results_header = ttk.Frame(right, style="Panel.TFrame")
        results_header.pack(fill=tk.X, pady=(8, 4))
        self.results_title_var = tk.StringVar(value="Results")
        ttk.Label(results_header, textvariable=self.results_title_var, style="PanelLabel.TLabel").pack(side="left")

        results_outer = ttk.Frame(right, style="Panel.TFrame")
        results_outer.pack(fill=tk.BOTH, expand=True)

        self.results_canvas = tk.Canvas(results_outer, highlightthickness=0, bg="#f8fafc")
        self.results_scroll = ttk.Scrollbar(results_outer, orient="vertical", command=self.results_canvas.yview)
        self.results_frame = ttk.Frame(self.results_canvas)

        self.results_frame.bind(
            "<Configure>",
            lambda _e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all")),
        )
        self.results_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.results_canvas.bind("<Configure>", self._on_results_canvas_resize)

        self.results_canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=self.results_scroll.set)

        self.results_canvas.pack(side="left", fill="both", expand=True)
        self.results_scroll.pack(side="right", fill="y")
        self._update_query_mode_state()

    def _on_mousewheel(self, event):
        self.results_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_results_canvas_resize(self, event):
        if event.width >= 1250:
            cards = 5
        elif event.width >= 1000:
            cards = 4
        elif event.width >= 760:
            cards = 3
        else:
            cards = 2

        if cards != self.cards_per_row:
            self.cards_per_row = cards
            if self.current_result_items:
                self._render_results(self.results_title_var.get().replace("Results for ", ""), self.current_result_items)

    def _update_query_mode_state(self):
        text_mode = self.query_mode.get() == "text"
        self.text_query_entry.configure(state=tk.NORMAL if text_mode else tk.DISABLED)
        self.image_query_entry.configure(state=tk.DISABLED if text_mode else tk.NORMAL)
        if not text_mode:
            self._refresh_query_preview()

    def browse_query_image(self):
        path = filedialog.askopenfilename(
            title="Choose query image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if path:
            self.image_query_var.set(path)
            self._refresh_query_preview()

    def _refresh_query_preview(self):
        self.preview_canvas.delete("all")
        query_path = Path(self.image_query_var.get())
        if not query_path.exists() or not query_path.is_file():
            self.preview_canvas.create_rectangle(0, 0, 250, 180, fill="#dde7ee", outline="#dde7ee")
            self.preview_canvas.create_text(
                125,
                90,
                text="Preview unavailable\n(image not found)",
                fill="#4b6578",
                font=("Segoe UI", 10),
                justify="center",
            )
            self.query_preview_ref = None
            return

        img = Image.open(query_path).convert("RGB")
        img.thumbnail((250, 180))
        self.query_preview_ref = ImageTk.PhotoImage(img)
        self.preview_canvas.create_rectangle(0, 0, 250, 180, fill="#dde7ee", outline="#dde7ee")
        self.preview_canvas.create_image(125, 90, image=self.query_preview_ref)

    def _set_busy(self, busy: bool):
        state = tk.DISABLED if busy else tk.NORMAL
        self.load_btn.configure(state=state)
        self.search_btn.configure(state=state)

    def _clear_results(self):
        for child in self.results_frame.winfo_children():
            child.destroy()
        self.image_refs = []

    def clear_results(self):
        self.current_result_items = []
        self._clear_results()
        self.results_title_var.set("Results")
        self.status_var.set("Results cleared")

    def _open_image_path(self, path: Path):
        if path.exists() and path.is_file():
            os.startfile(path)
        else:
            messagebox.showerror("Open image", f"Image not found: {path}")

    def load_index(self):
        self._run_background(self._load_index_worker, "Loading model and gallery index...")

    def _load_index_worker(self):
        try:
            cfg = load_config(self.config_var.get())
            checkpoint = Path(self.checkpoint_var.get())
            image_dir = Path(self.image_dir_var.get())
            cache_dir = Path(self.cache_dir_var.get())
            batch_size = int(self.batch_var.get())

            if not checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
            if not image_dir.exists() or not image_dir.is_dir():
                raise FileNotFoundError(f"Image folder not found: {image_dir}")

            requested_device = cfg.get("training", {}).get("device", "cpu")
            device = torch.device("cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu")

            image_encoder = ImageEncoder(
                embedding_dim=cfg["model"]["embedding_dim"],
                backbone_name=cfg["model"]["image_backbone"],
                use_pretrained=cfg["model"].get("image_pretrained", False),
            ).to(device)
            text_encoder = TextEncoder(
                embedding_dim=cfg["model"]["embedding_dim"],
                model_name=cfg["model"]["text_backbone"],
                use_pretrained=cfg["model"].get("text_pretrained", False),
            ).to(device)

            payload = torch.load(checkpoint, map_location=device)
            image_encoder.load_state_dict(payload["image_encoder"])
            text_encoder.load_state_dict(payload["text_encoder"])
            image_encoder.eval()
            text_encoder.eval()

            gallery_index, gallery_paths, used_cache_path = load_or_build_gallery_cache(
                image_encoder=image_encoder,
                image_dir=image_dir,
                checkpoint=checkpoint,
                embedding_dim=cfg["model"]["embedding_dim"],
                batch_size=batch_size,
                cache_dir=cache_dir,
                force_rebuild=self.rebuild_var.get(),
                device=device,
            )

            tokenizer = DistilBertTokenizer.from_pretrained(cfg["model"]["text_backbone"])

            self.cfg = cfg
            self.device = device
            self.image_encoder = image_encoder
            self.text_encoder = text_encoder
            self.gallery_index = gallery_index
            self.gallery_paths = gallery_paths
            self.cache_path = used_cache_path
            self.tokenizer = tokenizer

            self.root.after(
                0,
                lambda: self._stop_busy(
                    f"Loaded on {device}. Gallery images: {len(gallery_paths)}. Cache: {used_cache_path.with_suffix('.index')}"
                ),
            )
            self.root.after(
                0,
                lambda: self.info_var.set(
                    f"Device: {device}\nGallery: {image_dir}\nItems: {len(gallery_paths)}\nCache: {used_cache_path.with_suffix('.index')}"
                ),
            )
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Load failed", str(exc)))
            self.root.after(0, lambda: self._stop_busy("Failed to load index"))

    def search(self):
        if self.gallery_index is None:
            messagebox.showwarning("Not ready", "Load index first.")
            return

        self._run_background(self._search_worker, "Running search...")

    def _search_worker(self):
        try:
            mode = self.query_mode.get()
            top_k = int(self.topk_var.get())

            if mode == "text":
                text_query = self.text_query_var.get().strip()
                if not text_query:
                    raise ValueError("Please enter a text query.")
                query_embed = embed_text_query(
                    query=text_query,
                    text_encoder=self.text_encoder,
                    tokenizer=self.tokenizer,
                    max_length=self.cfg["data"].get("max_length", 77),
                    device=self.device,
                )
                query_label = f"text: {text_query}"
            else:
                query_image = Path(self.image_query_var.get())
                if not query_image.exists() or not query_image.is_file():
                    raise FileNotFoundError(f"Query image not found: {query_image}")
                query_embed = embed_image_query(
                    query_image=query_image,
                    image_encoder=self.image_encoder,
                    device=self.device,
                )
                query_label = f"image: {query_image}"

            scores, indices = topk_matches(query_embed, self.gallery_index, top_k)
            items = []
            for score, idx in zip(scores, indices):
                items.append((float(score), self.gallery_paths[idx]))

            self.root.after(0, lambda: self._render_results(query_label, items))
            self.root.after(0, lambda: self._stop_busy(f"Rendered {len(items)} matches"))
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Search failed", str(exc)))
            self.root.after(0, lambda: self._stop_busy("Search failed"))

    def _render_results(self, query_label: str, items):
        self._clear_results()
        self.current_result_items = items
        self.results_title_var.set(f"Results for {query_label}")

        for col in range(self.cards_per_row):
            self.results_frame.columnconfigure(col, weight=1)

        if not items:
            ttk.Label(
                self.results_frame,
                text="No matches found",
                style="PanelLabel.TLabel",
            ).grid(row=0, column=0, sticky="w", padx=12, pady=12)
            return

        thumb_size = (250, 250)

        for i, (score, path_str) in enumerate(items, start=1):
            row = (i - 1) // self.cards_per_row
            col = (i - 1) % self.cards_per_row

            card = ttk.Frame(self.results_frame, style="Card.TFrame", padding=8)
            card.grid(row=row, column=col, sticky="nsew", padx=7, pady=7)

            path = Path(path_str)
            pil_image = Image.open(path).convert("RGB")
            pil_image.thumbnail(thumb_size)
            tk_image = ImageTk.PhotoImage(pil_image)
            self.image_refs.append(tk_image)

            img_label = ttk.Label(card, image=tk_image)
            img_label.pack(pady=(0, 6))

            ttk.Label(card, text=f"#{i}    score={score:.4f}", style="PanelLabel.TLabel").pack(anchor="w")
            ttk.Label(card, text=path.name, style="PanelLabel.TLabel", wraplength=250).pack(anchor="w")
            ttk.Label(card, text=str(path), style="PanelLabel.TLabel", wraplength=250).pack(anchor="w", pady=(0, 6))

            ttk.Button(card, text="Open", command=lambda p=path: self._open_image_path(p)).pack(anchor="w")

        self.status_var.set(f"Rendered {len(items)} matches")


def main():
    root = tk.Tk()
    RetrievalUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
