

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import os

# ─────────────────────────────────────────────────────────────
# YOLO (carregado sob demanda para não travar a inicialização)
# ─────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[AVISO] ultralytics não instalada. Execute: pip install ultralytics")


# ══════════════════════════════════════════════════════════════
# MÓDULO DE PROCESSAMENTO DE IMAGEM
# ══════════════════════════════════════════════════════════════

class ImageProcessor:
    """Agrupa todas as funções de processamento das Etapas 1-5."""

    # ── ETAPA 2 ── Processamento básico ──────────────────────
    @staticmethod
    def to_gray(img_bgr):
        """Converte para escala de cinza."""
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def apply_blur(gray, ksize=5):
        """Aplica desfoque gaussiano."""
        ksize = ksize if ksize % 2 == 1 else ksize + 1   # kernel deve ser ímpar
        return cv2.GaussianBlur(gray, (ksize, ksize), 0)

    @staticmethod
    def detect_edges(blurred, low=100, high=200):
        """Detecta bordas com Canny."""
        return cv2.Canny(blurred, low, high)

    # ── ETAPA 3 ── Análise de cor / HSV ──────────────────────
    @staticmethod
    def to_hsv(img_bgr):
        """Converte para espaço de cor HSV."""
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    @staticmethod
    def shift_hue(img_bgr, delta_hue=30):
        """Altera o canal Hue (rotação de matiz)."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
        hsv[:, :, 0] = (hsv[:, :, 0] + delta_hue) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def boost_saturation(img_bgr, factor=1.5):
        """Multiplica a saturação pelo fator informado."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def split_hsv_channels(img_bgr):
        """Retorna (H, S, V) como arrays separados."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        return cv2.split(hsv)   # H, S, V

    # ── ETAPA 4 ── Histograma ─────────────────────────────────
    @staticmethod
    def compute_histogram(gray):
        """Calcula histograma de intensidade."""
        return cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

    @staticmethod
    def interpret_lighting(hist):
        """Retorna descrição textual da iluminação da imagem."""
        pixels = hist.sum()
        dark   = hist[:85].sum()  / pixels
        mid    = hist[85:170].sum() / pixels
        bright = hist[170:].sum() / pixels
        dominant = max([("Escura", dark), ("Média", mid), ("Clara", bright)],
                       key=lambda x: x[1])
        return (f"Iluminação predominante: {dominant[0]}  "
                f"| Escuro {dark:.0%}  Médio {mid:.0%}  Claro {bright:.0%}")

    # ── ETAPA 5 ── Binarização ────────────────────────────────
    @staticmethod
    def binarize(gray, thresh_val=127):
        """Threshold simples (Otsu se thresh_val=0)."""
        if thresh_val == 0:
            ret, binary = cv2.threshold(gray, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            ret, binary = cv2.threshold(gray, thresh_val, 255,
                                        cv2.THRESH_BINARY)
        return binary, ret

    @staticmethod
    def adaptive_threshold(gray):
        """Threshold adaptativo para iluminação não uniforme."""
        return cv2.adaptiveThreshold(gray, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)


# ══════════════════════════════════════════════════════════════
# MÓDULO DE IA – YOLO
# ══════════════════════════════════════════════════════════════

class AIDetector:
    """Encapsula YOLO para detecção e tracking (Etapa 6)."""

    def __init__(self, model_path="yolov8n.pt", conf=0.45):
        self.conf = conf
        self.model = None
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                print(f"[IA] Modelo YOLO carregado: {model_path}")
            except Exception as e:
                print(f"[IA] Erro ao carregar modelo: {e}")

    def detect(self, frame):
        """Detecção padrão. Retorna (imagem anotada, lista de labels, contagem)."""
        if self.model is None:
            return frame, [], 0
        results = self.model(frame, conf=self.conf, verbose=False)
        annotated = results[0].plot()
        labels = []
        if results[0].boxes is not None:
            for cls_id in results[0].boxes.cls.cpu().numpy():
                labels.append(self.model.names[int(cls_id)])
        return annotated, labels, len(labels)

    def track(self, frame):
        """Detecção + Tracking por ID persistente."""
        if self.model is None:
            return frame, []
        results = self.model.track(frame, conf=self.conf,
                                   persist=True, verbose=False)
        annotated = results[0].plot()
        ids = []
        if (results[0].boxes is not None and
                results[0].boxes.id is not None):
            ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()
        return annotated, ids

    def classify_image_type(self, labels):
        """Heurística simples para classificar o tipo de cena."""
        if not labels:
            return "Sem objetos detectados"
        counts = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1
        top = sorted(counts, key=counts.get, reverse=True)[:3]
        if "person" in top:
            return "Cena com pessoas"
        elif any(v in top for v in ["car", "truck", "bus", "motorcycle"]):
            return "Cena de trânsito / veículos"
        elif any(v in top for v in ["cat", "dog", "bird"]):
            return "Cena com animais"
        else:
            return f"Objetos: {', '.join(top)}"


# ══════════════════════════════════════════════════════════════
# INTERFACE GRÁFICA – TKINTER  (Desafio 1)
# ══════════════════════════════════════════════════════════════

class App(tk.Tk):
    """
    Janela principal com abas:
      • Imagem Estática  – etapas 1-7 em arquivo
      • Câmera ao Vivo   – tempo real + tracking (Desafios 2-4)
    """

    # ── cores do tema ──────────────────────────────────────────
    BG        = "#0d1117"
    PANEL     = "#161b22"
    ACCENT    = "#58a6ff"
    ACCENT2   = "#3fb950"
    DANGER    = "#f85149"
    FG        = "#c9d1d9"
    FG2       = "#8b949e"
    FONT_H    = ("Consolas", 11, "bold")
    FONT_B    = ("Consolas", 10)

    def __init__(self):
        super().__init__()
        self.title("🧠 Sistema Inteligente de Visão Computacional com IA")
        self.configure(bg=self.BG)
        self.geometry("1280x820")
        self.resizable(True, True)

        self.proc   = ImageProcessor()
        self.ai     = AIDetector()

        # estado
        self.current_bgr   = None   # imagem carregada (BGR)
        self.running        = False  # loop da câmera
        self.tracking_mode  = tk.BooleanVar(value=False)
        self.conf_var       = tk.DoubleVar(value=0.45)
        self.thresh_var     = tk.IntVar(value=127)
        self.hue_var        = tk.IntVar(value=30)
        self.sat_var        = tk.DoubleVar(value=1.5)
        self.blur_var       = tk.IntVar(value=5)

        self._build_ui()

    # ──────────────────────────────────────────────────────────
    # CONSTRUÇÃO DA UI
    # ──────────────────────────────────────────────────────────
    def _build_ui(self):
        # título
        header = tk.Frame(self, bg=self.BG)
        header.pack(fill="x", padx=20, pady=(14, 4))
        tk.Label(header, text="⬡  SISTEMA DE VISÃO COMPUTACIONAL COM IA",
                 font=("Consolas", 14, "bold"),
                 bg=self.BG, fg=self.ACCENT).pack(side="left")
        tk.Label(header, text="YOLOv8 · OpenCV · Tkinter",
                 font=self.FONT_B, bg=self.BG, fg=self.FG2).pack(side="right")

        sep = tk.Frame(self, bg=self.ACCENT, height=1)
        sep.pack(fill="x", padx=20, pady=4)

        # notebook (abas)
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TNotebook",       background=self.BG, borderwidth=0)
        style.configure("TNotebook.Tab",   background=self.PANEL, foreground=self.FG2,
                        font=self.FONT_H, padding=[14, 6])
        style.map("TNotebook.Tab",
                  background=[("selected", self.BG)],
                  foreground=[("selected", self.ACCENT)])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=12, pady=8)

        self.tab_static = tk.Frame(nb, bg=self.BG)
        self.tab_live   = tk.Frame(nb, bg=self.BG)
        nb.add(self.tab_static, text="📷  Imagem Estática")
        nb.add(self.tab_live,   text="🎥  Câmera ao Vivo")

        self._build_static_tab()
        self._build_live_tab()

    # ──────────────────────────────────────────────────────────
    # ABA: IMAGEM ESTÁTICA
    # ──────────────────────────────────────────────────────────
    def _build_static_tab(self):
        root = self.tab_static

        # ── painel esquerdo (controles) ────────────────────────
        left = tk.Frame(root, bg=self.PANEL, width=260)
        left.pack(side="left", fill="y", padx=(8, 4), pady=8)
        left.pack_propagate(False)

        self._section(left, "ETAPA 1 – Aquisição")
        tk.Button(left, text="📂  Abrir imagem",
                  command=self._load_image,
                  **self._btn_style(self.ACCENT)).pack(fill="x", padx=10, pady=4)

        self._section(left, "ETAPA 2 – Processamento")
        self._slider(left, "Blur (kernel)", self.blur_var, 1, 21)

        self._section(left, "ETAPA 3 – Cor / HSV")
        self._slider(left, "Δ Hue", self.hue_var, -90, 90)
        self._slider(left, "Saturação ×", self.sat_var, 0.0, 3.0, res=0.1)

        self._section(left, "ETAPA 5 – Binarização")
        self._slider(left, "Threshold (0=Otsu)", self.thresh_var, 0, 255)

        self._section(left, "ETAPA 6 – IA YOLO")
        self._slider(left, "Confiança mínima", self.conf_var, 0.1, 1.0, res=0.05)

        tk.Button(left, text="▶  Executar pipeline completo",
                  command=self._run_pipeline,
                  **self._btn_style(self.ACCENT2)).pack(fill="x", padx=10, pady=(12, 4))

        # status
        self.status_lbl = tk.Label(left, text="Aguardando imagem…",
                                   font=self.FONT_B, bg=self.PANEL,
                                   fg=self.FG2, wraplength=230, justify="left")
        self.status_lbl.pack(padx=10, pady=6)

        # ── painel direito (canvas / resultado) ───────────────
        right = tk.Frame(root, bg=self.BG)
        right.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)

        # grade 3×2 de imagens
        self.canvases = {}
        titles = ["Original", "Escala de Cinza", "Bordas (Canny)",
                  "HSV – Hue shift", "Binarização", "IA – Detecção YOLO"]
        grid = tk.Frame(right, bg=self.BG)
        grid.pack(fill="both", expand=True)

        for i, title in enumerate(titles):
            r, c = divmod(i, 3)
            cell = tk.Frame(grid, bg=self.PANEL,
                            highlightbackground=self.ACCENT,
                            highlightthickness=1)
            cell.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
            grid.rowconfigure(r, weight=1)
            grid.columnconfigure(c, weight=1)

            tk.Label(cell, text=title, font=self.FONT_B,
                     bg=self.PANEL, fg=self.ACCENT2).pack(pady=(6, 2))
            cnv = tk.Label(cell, bg=self.PANEL)
            cnv.pack(fill="both", expand=True, padx=4, pady=(0, 6))
            self.canvases[title] = cnv

        # histograma embarcado
        self._section(right, "ETAPA 4 – Histograma de Intensidade")
        self.hist_frame = tk.Frame(right, bg=self.BG)
        self.hist_frame.pack(fill="x", padx=8, pady=4)

        # resultado textual
        self.result_text = tk.Text(right, height=4, bg=self.PANEL,
                                   fg=self.FG, font=self.FONT_B,
                                   relief="flat", bd=0, state="disabled")
        self.result_text.pack(fill="x", padx=8, pady=4)

    # ──────────────────────────────────────────────────────────
    # ABA: CÂMERA AO VIVO
    # ──────────────────────────────────────────────────────────
    def _build_live_tab(self):
        root = self.tab_live

        # controles
        ctrl = tk.Frame(root, bg=self.PANEL, width=220)
        ctrl.pack(side="left", fill="y", padx=(8, 4), pady=8)
        ctrl.pack_propagate(False)

        self._section(ctrl, "CÂMERA AO VIVO")
        tk.Button(ctrl, text="▶  Iniciar",
                  command=self._start_camera,
                  **self._btn_style(self.ACCENT2)).pack(fill="x", padx=10, pady=4)
        tk.Button(ctrl, text="⏹  Parar",
                  command=self._stop_camera,
                  **self._btn_style(self.DANGER)).pack(fill="x", padx=10, pady=4)

        self._section(ctrl, "MODO")
        tk.Checkbutton(ctrl, text="🔁 Tracking por ID",
                       variable=self.tracking_mode,
                       bg=self.PANEL, fg=self.FG,
                       selectcolor=self.BG,
                       activebackground=self.PANEL,
                       font=self.FONT_B).pack(anchor="w", padx=12)

        self._slider(ctrl, "Confiança", self.conf_var, 0.1, 1.0, res=0.05)

        self._section(ctrl, "MÉTRICAS AO VIVO")
        self.fps_lbl    = self._metric_label(ctrl, "FPS", "—")
        self.obj_lbl    = self._metric_label(ctrl, "Objetos/frame", "—")
        self.total_lbl  = self._metric_label(ctrl, "Total acumulado", "—")
        self.scene_lbl  = self._metric_label(ctrl, "Tipo de cena", "—")

        # canvas principal
        right = tk.Frame(root, bg=self.BG)
        right.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=8)

        self.live_canvas = tk.Label(right, bg="#000000",
                                    text="Câmera não iniciada",
                                    fg=self.FG2, font=self.FONT_H)
        self.live_canvas.pack(fill="both", expand=True)

        # log de alertas
        self._section(right, "LOG DE ALERTAS")
        self.alert_box = tk.Text(right, height=4, bg=self.PANEL,
                                 fg=self.DANGER, font=self.FONT_B,
                                 relief="flat", bd=0, state="disabled")
        self.alert_box.pack(fill="x", padx=8, pady=(0, 6))

    # ──────────────────────────────────────────────────────────
    # UTILITÁRIOS DE UI
    # ──────────────────────────────────────────────────────────
    def _btn_style(self, color):
        return dict(bg=color, fg="#000000", font=self.FONT_H,
                    activebackground=color, relief="flat",
                    cursor="hand2", pady=6)

    def _section(self, parent, text):
        f = tk.Frame(parent, bg=parent["bg"])
        f.pack(fill="x", padx=8, pady=(10, 2))
        tk.Label(f, text=text, font=("Consolas", 9, "bold"),
                 bg=parent["bg"], fg=self.FG2).pack(anchor="w")
        tk.Frame(f, bg=self.FG2, height=1).pack(fill="x", pady=(2, 0))

    def _slider(self, parent, label, var, from_, to, res=1):
        frm = tk.Frame(parent, bg=parent["bg"])
        frm.pack(fill="x", padx=10, pady=2)
        tk.Label(frm, text=label, font=self.FONT_B,
                 bg=parent["bg"], fg=self.FG).pack(anchor="w")
        row = tk.Frame(frm, bg=parent["bg"])
        row.pack(fill="x")
        s = tk.Scale(row, variable=var, from_=from_, to=to, resolution=res,
                     orient="horizontal", bg=parent["bg"], fg=self.FG,
                     highlightthickness=0, troughcolor=self.BG,
                     activebackground=self.ACCENT, length=160, showvalue=True)
        s.pack(side="left")

    def _metric_label(self, parent, key, val):
        frm = tk.Frame(parent, bg=self.PANEL)
        frm.pack(fill="x", padx=10, pady=2)
        tk.Label(frm, text=key + ":", font=self.FONT_B,
                 bg=self.PANEL, fg=self.FG2, width=18, anchor="w").pack(side="left")
        lbl = tk.Label(frm, text=val, font=self.FONT_H,
                       bg=self.PANEL, fg=self.ACCENT)
        lbl.pack(side="left")
        return lbl

    def _show_on_canvas(self, key, img_bgr_or_gray):
        """Redimensiona e exibe imagem num Label de canvas."""
        cnv = self.canvases.get(key)
        if cnv is None:
            return
        w, h = cnv.winfo_width() or 300, cnv.winfo_height() or 200
        w, h = max(w, 120), max(h, 90)
        if len(img_bgr_or_gray.shape) == 2:
            img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb).resize((w, h), Image.LANCZOS)
        imgtk = ImageTk.PhotoImage(pil)
        cnv.imgtk = imgtk
        cnv.config(image=imgtk)

    def _write_result(self, text):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", "end")
        self.result_text.insert("end", text)
        self.result_text.config(state="disabled")

    def _log_alert(self, msg):
        self.alert_box.config(state="normal")
        self.alert_box.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.alert_box.see("end")
        self.alert_box.config(state="disabled")

    def _update_status(self, msg):
        self.status_lbl.config(text=msg)

    # ──────────────────────────────────────────────────────────
    # PIPELINE ESTÁTICO (Etapas 1-7)
    # ──────────────────────────────────────────────────────────
    def _load_image(self):
        """ETAPA 1 – Aquisição."""
        path = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Erro", "Não foi possível carregar a imagem.")
            return
        self.current_bgr = img
        self._update_status(f"Imagem: {os.path.basename(path)}\n"
                            f"Tamanho: {img.shape[1]}×{img.shape[0]} px")
        self._show_on_canvas("Original", img)

    def _run_pipeline(self):
        """Executa etapas 2-7 na imagem carregada."""
        if self.current_bgr is None:
            messagebox.showwarning("Atenção", "Abra uma imagem primeiro (Etapa 1).")
            return
        img = self.current_bgr.copy()
        self.ai.conf = self.conf_var.get()

        # ── ETAPA 2 – Processamento ───────────────────────────
        gray   = self.proc.to_gray(img)
        blur   = self.proc.apply_blur(gray, self.blur_var.get())
        edges  = self.proc.detect_edges(blur)
        self._show_on_canvas("Escala de Cinza", gray)
        self._show_on_canvas("Bordas (Canny)", edges)

        # ── ETAPA 3 – HSV / Cor ───────────────────────────────
        hue_shifted = self.proc.shift_hue(img, self.hue_var.get())
        sat_boosted = self.proc.boost_saturation(img, self.sat_var.get())
        H, S, V     = self.proc.split_hsv_channels(img)
        self._show_on_canvas("HSV – Hue shift", hue_shifted)

        # ── ETAPA 4 – Histograma ──────────────────────────────
        hist  = self.proc.compute_histogram(gray)
        light = self.proc.interpret_lighting(hist)
        self._draw_histogram(hist, light)

        # ── ETAPA 5 – Binarização ─────────────────────────────
        binary, tval = self.proc.binarize(gray, self.thresh_var.get())
        self._show_on_canvas("Binarização", binary)

        # ── ETAPA 6 – IA (YOLO) ───────────────────────────────
        annotated, labels, count = self.ai.detect(img)
        self._show_on_canvas("IA – Detecção YOLO", annotated)

        # ── ETAPA 7 – Resultado final ─────────────────────────
        scene  = self.ai.classify_image_type(labels)
        unique = list(dict.fromkeys(labels))
        info = (
            f"═══ RESULTADO FINAL ═══\n"
            f"Objetos detectados ({count}): {', '.join(unique) if unique else 'nenhum'}\n"
            f"Tipo de cena: {scene}\n"
            f"{light}\n"
            f"Threshold aplicado: {tval}"
        )
        self._write_result(info)
        self._update_status("Pipeline concluído ✓")

    def _draw_histogram(self, hist, light_text):
        """Renderiza histograma no frame embarcado."""
        for w in self.hist_frame.winfo_children():
            w.destroy()
        fig, ax = plt.subplots(figsize=(9, 1.6), facecolor=self.BG)
        ax.set_facecolor(self.PANEL)
        x = np.arange(256)
        ax.fill_between(x, hist, color=self.ACCENT, alpha=0.7)
        ax.plot(x, hist, color=self.ACCENT, linewidth=0.8)
        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensidade", color=self.FG2, fontsize=8)
        ax.set_ylabel("Frequência", color=self.FG2, fontsize=8)
        ax.tick_params(colors=self.FG2, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(self.FG2)
        ax.set_title(light_text, color=self.ACCENT2, fontsize=8)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.hist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x")
        plt.close(fig)

    # ──────────────────────────────────────────────────────────
    # CÂMERA AO VIVO (Desafios 2-4)
    # ──────────────────────────────────────────────────────────
    def _start_camera(self):
        if self.running:
            return
        self.running    = True
        self.total_objs = 0
        t = threading.Thread(target=self._camera_loop, daemon=True)
        t.start()

    def _stop_camera(self):
        self.running = False

    def _camera_loop(self):
        """Loop de captura + detecção em thread separada."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._log_alert("Webcam não encontrada. Verifique a conexão.")
            self.running = False
            return

        ALERT_THRESHOLD = 3
        frame_count     = 0
        t0              = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            self.ai.conf = self.conf_var.get()

            # ─ detecção ou tracking ───────────────────────────
            if self.tracking_mode.get():
                annotated, ids = self.ai.track(frame)
                count  = len(ids)
                labels = []
            else:
                annotated, labels, count = self.ai.detect(frame)

            self.total_objs += count
            frame_count     += 1
            elapsed          = time.time() - t0
            fps              = frame_count / elapsed if elapsed > 0 else 0

            # ─ overlay de texto ───────────────────────────────
            overlay = annotated.copy()
            cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (88, 166, 255), 2)
            cv2.putText(overlay, f"Objetos: {count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (63, 185, 80), 2)
            cv2.putText(overlay, f"Total: {self.total_objs}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            if count > ALERT_THRESHOLD:
                cv2.putText(overlay, "⚠ ALERTA: Múltiplos objetos!", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (248, 81, 73), 3)
                self.after(0, self._log_alert,
                           f"Alerta! {count} objetos no frame.")

            # ─ atualiza UI na thread principal ───────────────
            self.after(0, self._update_live_metrics,
                       fps, count, self.total_objs,
                       self.ai.classify_image_type(labels))
            self.after(0, self._update_live_canvas, overlay)

            # Limita a ~30 fps
            time.sleep(max(0, 1/30 - (time.time() - t0 - (frame_count - 1) / 30)))

        cap.release()
        self.after(0, self.live_canvas.config,
                   {"image": "", "text": "Câmera encerrada"})

    def _update_live_canvas(self, frame_bgr):
        w = self.live_canvas.winfo_width()  or 800
        h = self.live_canvas.winfo_height() or 560
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil     = Image.fromarray(img_rgb).resize((w, h), Image.LANCZOS)
        imgtk   = ImageTk.PhotoImage(pil)
        self.live_canvas.imgtk = imgtk
        self.live_canvas.config(image=imgtk, text="")

    def _update_live_metrics(self, fps, count, total, scene):
        self.fps_lbl.config(text=f"{fps:.1f}")
        self.obj_lbl.config(text=str(count))
        self.total_lbl.config(text=str(total))
        self.scene_lbl.config(text=scene)


# ══════════════════════════════════════════════════════════════
# MODO HEADLESS (sem interface) – para ambientes sem display
# ══════════════════════════════════════════════════════════════

def run_headless(image_path: str):
    """
    Executa todas as etapas sem interface gráfica.
    Salva os resultados como arquivos de imagem.
    """
    print("=" * 60)
    print("SISTEMA DE VISÃO COMPUTACIONAL - MODO HEADLESS")
    print("=" * 60)

    proc = ImageProcessor()
    ai   = AIDetector()

    # ETAPA 1
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERRO] Imagem não encontrada: {image_path}")
        return
    print(f"[OK] Imagem carregada: {img.shape[1]}×{img.shape[0]} px")

    # ETAPA 2
    gray  = proc.to_gray(img)
    blur  = proc.apply_blur(gray, 5)
    edges = proc.detect_edges(blur)
    cv2.imwrite("saida_gray.jpg", gray)
    cv2.imwrite("saida_edges.jpg", edges)
    print("[OK] Etapa 2 – Escala de cinza e bordas salvas.")

    # ETAPA 3
    hsv         = proc.to_hsv(img)
    hue_shifted = proc.shift_hue(img, 30)
    sat_boosted = proc.boost_saturation(img, 1.5)
    cv2.imwrite("saida_hue_shift.jpg", hue_shifted)
    cv2.imwrite("saida_sat_boost.jpg", sat_boosted)
    print("[OK] Etapa 3 – Variações HSV salvas.")

    # ETAPA 4
    hist  = proc.compute_histogram(gray)
    light = proc.interpret_lighting(hist)
    print(f"[OK] Etapa 4 – {light}")
    plt.figure(figsize=(10, 3))
    plt.fill_between(range(256), hist, alpha=0.7)
    plt.title(light)
    plt.savefig("saida_histograma.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ETAPA 5
    binary, tval = proc.binarize(gray)
    cv2.imwrite("saida_binarizado.jpg", binary)
    print(f"[OK] Etapa 5 – Binarizado com threshold={tval:.0f}")

    # ETAPA 6 & 7
    annotated, labels, count = ai.detect(img)
    cv2.imwrite("saida_ia_deteccao.jpg", annotated)
    scene = ai.classify_image_type(labels)
    print(f"[OK] Etapa 6-7 – {count} objetos: {labels}")
    print(f"     Tipo de cena: {scene}")
    print("=" * 60)
    print("Arquivos salvos no diretório atual.")


# ══════════════════════════════════════════════════════════════
# PONTO DE ENTRADA
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Modo headless: python sistema_visao_ia.py imagem.jpg
        run_headless(sys.argv[1])
    else:
        # Modo GUI
        app = App()
        app.mainloop()
