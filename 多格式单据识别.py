import sys
import os
import re
import shutil
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, DoubleVar, PhotoImage
from PIL import Image, ImageTk
import threading
import time
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== 核心：强制获取程序根目录 ==========
if getattr(sys, 'frozen', False):
    EXE_DIR = os.path.dirname(os.path.abspath(sys.executable))
    BASE_DIR = os.path.join(EXE_DIR, "_internal")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== 安全的高DPI适配 ==========
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

# ========== 导入核心库 ==========
import cv2
import onnxruntime as ort
ort.set_default_logger_severity(4)
os.environ["OMP_NUM_THREADS"] = "1"
providers = ['CPUExecutionProvider']

# PyMuPDF
try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        messagebox.showerror("错误", "请安装PyMuPDF：pip install PyMuPDF")
        sys.exit(1)

# OCR引擎
try:
    from onnxocr.onnx_paddleocr import ONNXPaddleOcr
except ImportError:
    messagebox.showerror("错误", "未找到onnxocr模块！")
    sys.exit(1)

# ===================== 全局配置 =====================
DEFAULT_PDF_DIR = r"d:/Users/Administrator/Desktop/PDF待识别"
CLS_MODEL_PATH = None
USE_GPU = False
TEMP_IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, 'frozen', False) else BASE_DIR, "temp_pdf_imgs")

# 默认格式：字母-数字
DEFAULT_FORMAT = "字母-数字"

# 识别字符集
TARGET_CHARS = "0123456789AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz"

# ===================== 格式转换：用户输入 → 自动生成正则 =====================
def user_format_to_regex(user_input):
    pattern = user_input.strip()
    pattern = pattern.replace("数字", r"(\d{1,10})")
    pattern = pattern.replace("字母", r"([A-Za-z]{1,10})")
    pattern = pattern.replace("Y", r"([Yy])")
    pattern = pattern.replace("Z", r"([Zz])")
    pattern = pattern.replace("2", r"([2])")
    pattern = pattern.replace("-", r"[-_ .]?")
    pattern = pattern.replace("_", r"[-_ .]?")
    return pattern

def extract_target_pattern(text, user_format):
    try:
        regex_str = user_format_to_regex(user_format)
        pattern = re.compile(regex_str, re.I)
        matches = pattern.findall(text)
        if matches:
            parts = matches[0]
            if isinstance(parts, str):
                return parts.strip().upper()
            clean_parts = [str(p).strip().upper() for p in parts]
            return "-".join(clean_parts)
    except Exception as e:
        pass
    return None

# ===================== 核心功能函数 =====================
def pdf_extract_first_page_upper(pdf_path, ratio=0.25):
    try:
        pdf_doc = fitz.open(pdf_path)
        page = pdf_doc[0]
        dpi = 120
        page_width = page.rect.width
        page_height = page.rect.height
        upper_rect = fitz.Rect(0, 0, page_width, page_height * ratio)
        pix = page.get_pixmap(dpi=dpi, clip=upper_rect, alpha=False, annots=True)
        pdf_doc.close()
        if pix is None or pix.width == 0 or pix.height == 0:
            return None, f"提取{ratio*100}%区域为空"
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img, ""
    except Exception as e:
        return None, f"提取失败：{str(e)}"

def pdf_extract_first_page_upper_disk(pdf_path, output_dir, ratio=0.25):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        pdf_doc = fitz.open(pdf_path)
        page = pdf_doc[0]
        dpi = 150
        page_width = page.rect.width
        page_height = page.rect.height
        upper_rect = fitz.Rect(0, 0, page_width, page_height * ratio)
        pix = page.get_pixmap(dpi=dpi, clip=upper_rect, alpha=False, annots=True)
        pdf_doc.close()
        if pix is None or pix.width == 0 or pix.height == 0:
            return None, "区域为空"
        img_filename = f"preview_{os.path.basename(pdf_path)}_{ratio}.jpg"
        img_path = os.path.join(output_dir, img_filename)
        pix.save(img_path)
        return img_path, ""
    except Exception as e:
        return None, str(e)

def enhance_contrast_only(img, alpha=1.8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.convertScaleAbs(gray, alpha=alpha, beta=0, dst=gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def extract_from_img(img_or_path, ocr_engine, user_format, contrast_alpha=1.8):
    if isinstance(img_or_path, np.ndarray):
        img = img_or_path
    else:
        if not os.path.exists(img_or_path):
            return None, "路径无效"
        img = cv2.imread(img_or_path)
    if img is None:
        return None, "图片为空"

    max_size = 1200
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

    enhanced_img = enhance_contrast_only(img, alpha=contrast_alpha)
    ocr_result = ocr_engine.ocr(enhanced_img, cls=False)
    del img, enhanced_img
    gc.collect()

    all_text = ""
    for line in ocr_result:
        def get_text(obj):
            if isinstance(obj, str):
                return obj.strip()
            if isinstance(obj, (list, tuple)):
                for item in obj:
                    t = get_text(item)
                    if t: return t
            return ""
        t = get_text(line)
        if t: all_text += t + " "

    target = extract_target_pattern(all_text, user_format)
    if target:
        return target, f"识别成功：{target}"
    return None, f"未匹配格式，文本：{all_text[:80]}"

def ocr_extract_target_upper(pdf_path, ocr_engine, user_format, contrast_alpha=1.8):
    for ratio in [0.25, 0.5, 1.0]:
        img, err = pdf_extract_first_page_upper(pdf_path, ratio)
        if img is not None:
            target, err2 = extract_from_img(img, ocr_engine, user_format, contrast_alpha)
            if target:
                return target, f"{int(ratio*100)}%区域识别成功"
    return None, "所有区域均未识别到目标格式"

def rename_to_pure_target(original_pdf_path, target_str):
    pdf_dir = os.path.dirname(original_pdf_path)
    new_filename = f"{target_str}.pdf"
    new_path = os.path.join(pdf_dir, new_filename)
    counter = 2
    while os.path.exists(new_path):
        new_filename = f"{target_str}_{counter}.pdf"
        new_path = os.path.join(pdf_dir, new_filename)
        counter += 1
    shutil.move(original_pdf_path, new_path)
    return new_filename

# ===================== GUI =====================
class PDFOCRRenamerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF单据识别重命名工具")
        self.root.geometry("1250x820")
        self.root.resizable(True, True)

        self.ocr_engine = None
        self._ocr_params = None
        self.is_running = False
        self.loading_animation = None
        self._log_buffer = []
        self.ocr_type = "print"

        self.create_style()
        self.create_widgets()
        self.init_ocr_engine()

    def create_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background="#f5f5f5", foreground="#333", font=("Microsoft YaHei UI", 10))
        style.configure("Primary.TButton", background="#165DFF", foreground="white", font=("Microsoft YaHei UI",11,"bold"))
        style.map("Primary.TButton", background=[("active","#0E42D2"),("disabled","#99B8FF")])
        style.configure("Secondary.TButton", background="#6B7280", foreground="white")
        style.configure("Card.TLabelframe", foreground="#165DFF", font=("Microsoft YaHei UI",12,"bold"))

    def create_widgets(self):
        # 标题
        title_frame = tk.Frame(self.root, bg="#165DFF", height=70)
        title_frame.pack(fill=tk.X)
        tk.Label(title_frame, text="PDF单据识别重命名工具", font=("Microsoft YaHei UI",16,"bold"),
                 bg="#165DFF", fg="white").pack(expand=True)

        # 主布局
        main = tk.Frame(self.root, bg="#f5f5f5")
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        left = tk.Frame(main, bg="#f5f5f5")
        left.grid(row=0, column=0, sticky="nsew", padx=(0,20))
        left.grid_rowconfigure(6, weight=1)
        left.grid_columnconfigure(0, weight=1)

        right = tk.Frame(main, bg="#f5f5f5")
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # ========== 1 目录 ==========
        dir_card = ttk.LabelFrame(left, text="目录设置", style="Card.TLabelframe")
        dir_card.grid(row=0, column=0, sticky="ew", pady=(0,15))
        dir_inner = tk.Frame(dir_card, bg="white")
        dir_inner.grid(row=0, column=0, sticky="ew", padx=15, pady=15)
        dir_inner.grid_columnconfigure(1, weight=1)
        self.var_pdf_dir = tk.StringVar(value=DEFAULT_PDF_DIR)
        tk.Label(dir_inner, text="目录：", bg="white").grid(row=0,column=0,sticky="w")
        ttk.Entry(dir_inner, textvariable=self.var_pdf_dir).grid(row=0,column=1,sticky="ew",padx=10)
        ttk.Button(dir_inner, text="浏览", command=self.browse_pdf_dir, style="Secondary.TButton").grid(row=0,column=2)

        # ========== 2 格式输入 ==========
        pattern_card = ttk.LabelFrame(left, text="识别格式（支持：字母-数字 / 数字-字母 等）", style="Card.TLabelframe")
        pattern_card.grid(row=1, column=0, sticky="ew", pady=(0,15))
        pattern_inner = tk.Frame(pattern_card, bg="white")
        pattern_inner.grid(row=0, column=0, sticky="ew", padx=15, pady=15)
        pattern_inner.grid_columnconfigure(0, weight=1)
        self.var_format = tk.StringVar(value=DEFAULT_FORMAT)
        tk.Label(pattern_inner, text="识别格式：", bg="white").grid(row=0,column=0,sticky="w")
        ttk.Entry(pattern_inner, textvariable=self.var_format).grid(row=1,column=0,sticky="ew",pady=5)
        tk.Label(pattern_inner, text='示例：字母-数字  →  A-123、B_456、X 789 都能识别',
                 bg="white", fg="#0066CC", font=("",9)).grid(row=2,column=0,sticky="w")

        # ========== 3 OCR 类型 ==========
        ocr_type_card = ttk.LabelFrame(left, text="OCR识别类型", style="Card.TLabelframe")
        ocr_type_card.grid(row=2, column=0, sticky="ew", pady=(0,15))
        ocr_type_inner = tk.Frame(ocr_type_card, bg="white")
        ocr_type_inner.grid(row=0, column=0, sticky="ew", padx=15, pady=15)
        self.var_ocr_type = tk.StringVar(value="print")
        ttk.Radiobutton(ocr_type_inner, text="📄 印刷体", variable=self.var_ocr_type,
                        value="print", command=self.on_ocr_type_change).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(ocr_type_inner, text="✍️ 手写体", variable=self.var_ocr_type,
                        value="handwrite", command=self.on_ocr_type_change).pack(side=tk.LEFT, padx=10)

        # ========== 4 参数 ==========
        param_card = ttk.LabelFrame(left, text="识别参数", style="Card.TLabelframe")
        param_card.grid(row=3, column=0, sticky="ew", pady=(0,15))
        param_inner = tk.Frame(param_card, bg="white")
        param_inner.grid(row=0, column=0, sticky="ew", padx=15, pady=15)
        param_inner.grid_columnconfigure(2, weight=1)

        self.contrast_alpha_var = DoubleVar(value=1.8)
        tk.Label(param_inner, text="对比度：", bg="white").grid(row=0,column=0,sticky="w")
        ttk.Entry(param_inner, textvariable=self.contrast_alpha_var, width=8).grid(row=0,column=1,padx=5)
        ttk.Scale(param_inner, from_=0.1, to=5, variable=self.contrast_alpha_var).grid(row=0,column=2,sticky="ew",padx=5)

        self.det_thresh_var = DoubleVar(value=0.15)
        tk.Label(param_inner, text="检测阈值：", bg="white").grid(row=1,column=0,sticky="w")
        ttk.Entry(param_inner, textvariable=self.det_thresh_var, width=8).grid(row=1,column=1,padx=5)
        ttk.Scale(param_inner, from_=0, to=1, variable=self.det_thresh_var).grid(row=1,column=2,sticky="ew",padx=5)

        self.rec_thresh_var = DoubleVar(value=0.4)
        tk.Label(param_inner, text="识别阈值：", bg="white").grid(row=2,column=0,sticky="w")
        ttk.Entry(param_inner, textvariable=self.rec_thresh_var, width=8).grid(row=2,column=1,padx=5)
        ttk.Scale(param_inner, from_=0, to=1, variable=self.rec_thresh_var).grid(row=2,column=2,sticky="ew",padx=5)

        ttk.Button(param_inner, text="应用参数并重启OCR", command=self.apply_conf_params, style="Primary.TButton")\
            .grid(row=3,column=0,columnspan=3,sticky="ew",pady=8)

        # ========== 5 预览 ==========
        prev_card = ttk.LabelFrame(left, text="预览", style="Card.TLabelframe")
        prev_card.grid(row=4, column=0, sticky="ew", pady=(0,15))
        prev_inner = tk.Frame(prev_card, bg="white")
        prev_inner.grid(row=0, column=0, sticky="ew", padx=15, pady=15)
        prev_inner.grid_columnconfigure(1, weight=1)
        self.preview_pdf_var = tk.StringVar(value="未选择")
        tk.Label(prev_inner, text="文件：", bg="white").grid(row=0,column=0,sticky="w")
        ttk.Entry(prev_inner, textvariable=self.preview_pdf_var, state="readonly").grid(row=0,column=1,sticky="ew",padx=10)
        ttk.Button(prev_inner, text="选择", command=self.select_preview_pdf, style="Secondary.TButton").grid(row=0,column=2)
        ttk.Button(prev_inner, text="预览", command=self.preview_contrast_effect, style="Primary.TButton").grid(row=0,column=3)

        # ========== 6 控制 ==========
        ctrl_card = ttk.LabelFrame(left, text="批量处理", style="Card.TLabelframe")
        ctrl_card.grid(row=5, column=0, sticky="ew", pady=(0,15))
        ctrl_inner = tk.Frame(ctrl_card, bg="white")
        ctrl_inner.grid(row=0, column=0, sticky="ew", padx=15, pady=15)
        ctrl_inner.grid_columnconfigure(0, weight=1)
        self.btn_start = ttk.Button(ctrl_inner, text="开始处理", command=self.start_process, style="Primary.TButton")
        self.btn_start.grid(row=0,column=0,sticky="ew",padx=5)
        self.btn_stop = ttk.Button(ctrl_inner, text="停止", command=self.stop_process, state=tk.DISABLED, style="Secondary.TButton")
        self.btn_stop.grid(row=0,column=1,sticky="ew",padx=5)
        self.loading_label = tk.Label(ctrl_inner, text="", bg="white", fg="#165DFF", font=("",10,"bold"))
        self.loading_label.grid(row=0,column=2,padx=10)
        self.stat_var = tk.StringVar(value="成功：0 | 失败：0 | 总计：0")
        tk.Label(ctrl_inner, textvariable=self.stat_var, bg="white", font=("",10,"bold")).grid(row=1,column=0,columnspan=3,pady=5)

        # ========== 日志 ==========
        log_card = ttk.LabelFrame(left, text="日志", style="Card.TLabelframe")
        log_card.grid(row=6, column=0, sticky="nsew")
        log_card.grid_rowconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_card, font=("Consolas",9), bg="#f8f8f8")
        self.log_text.grid(row=0,column=0,sticky="nsew",padx=15,pady=15)

        # ========== 右侧预览 ==========
        img_card = ttk.LabelFrame(right, text="预览图", style="Card.TLabelframe")
        img_card.grid(row=0,column=0,sticky="nsew")
        img_card.grid_rowconfigure(0, weight=1)
        self.preview_img_label = tk.Label(img_card, bg="#f8f8f8", text="选择PDF预览")
        self.preview_img_label.grid(row=0,column=0,sticky="nsew",padx=15,pady=15)

    def on_ocr_type_change(self):
        self.ocr_type = self.var_ocr_type.get()
        self.log(f"切换OCR类型：{'手写体' if self.ocr_type=='handwrite' else '印刷体'}")

    def browse_pdf_dir(self):
        d = filedialog.askdirectory()
        if d: self.var_pdf_dir.set(d)

    def select_preview_pdf(self):
        f = filedialog.askopenfilename(filetypes=[("PDF","*.pdf")])
        if f:
            self.preview_pdf_path = f
            self.preview_pdf_var.set(os.path.basename(f))

    def preview_contrast_effect(self):
        if not hasattr(self,'preview_pdf_path'):
            messagebox.showwarning("提示","请先选择PDF")
            return
        try:
            p, err = pdf_extract_first_page_upper_disk(self.preview_pdf_path, TEMP_IMG_DIR, 0.25)
            if err:
                p, err = pdf_extract_first_page_upper_disk(self.preview_pdf_path, TEMP_IMG_DIR, 0.5)
            img = cv2.imread(p)
            enh = enhance_contrast_only(img, self.contrast_alpha_var.get())
            h,w = enh.shape[:2]
            scale = min(380/w, 450/h)
            resized = cv2.resize(enh, (int(w*scale), int(h*scale)))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tkimg = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.preview_img_label.config(image=tkimg, text="")
            self.preview_img_label.image = tkimg
            os.remove(p)
        except:
            messagebox.showerror("错误","预览失败")

    def init_ocr_engine(self, det_thresh=None, rec_thresh=None):
        det = det_thresh or self.det_thresh_var.get()
        rec = rec_thresh or self.rec_thresh_var.get()
        if hasattr(self,'_ocr_params') and self._ocr_params == (det, rec, self.ocr_type):
            self.log("OCR参数无变化，直接复用")
            return
        self._ocr_params = (det, rec, self.ocr_type)
        if self.ocr_engine: del self.ocr_engine; gc.collect()

        if getattr(sys,'frozen',False):
            root = os.path.dirname(sys.executable)
            base = os.path.join(root, "_internal", "onnxocr", "models")
        else:
            base = os.path.join(BASE_DIR, "onnxocr", "models")

        if self.ocr_type == "handwrite":
            det_path = os.path.join(base, "handwrite", "det.onnx")
            rec_path = os.path.join(base, "handwrite", "rec.onnx")
        else:
            det_path = os.path.join(base, "ppocrv5", "det", "det.onnx")
            rec_path = os.path.join(base, "ppocrv5", "rec", "rec.onnx")

        try:
            self.ocr_engine = ONNXPaddleOcr(
                det_model=det_path, rec_model=rec_path, cls_model=CLS_MODEL_PATH,
                use_gpu=USE_GPU, use_angle_cls=False, det_thresh=det, rec_thresh=rec,
                det_db_unclip_ratio=1.6, use_dilation=True, rec_char_dict=TARGET_CHARS, providers=providers)
            self.log(f"✅ OCR初始化完成：{'手写体' if self.ocr_type=='handwrite' else '印刷体'}")
        except Exception as e:
            self.log(f"OCR失败：{str(e)}")
            messagebox.showerror("错误", f"OCR加载失败：{str(e)}")

    def apply_conf_params(self):
        self.init_ocr_engine()
        self.log("参数已应用")

    def start_process(self):
        if self.is_running: return
        if not self.ocr_engine:
            messagebox.showwarning("提示","请先初始化OCR")
            return
        d = self.var_pdf_dir.get()
        if not os.path.exists(d):
            messagebox.showerror("错误","目录不存在")
            return
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.update_loading()
        threading.Thread(target=self.run_batch, daemon=True).start()

    def stop_process(self):
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

    def update_loading(self):
        if not self.is_running: return
        txt = self.loading_label["text"]
        self.loading_label.config(text="处理中.." if txt=="处理中..." else "处理中...")
        self.loading_animation = self.root.after(500, self.update_loading)

    def _process_one(self, path):
        try:
            fmt = self.var_format.get()
            alpha = self.contrast_alpha_var.get()
            t, err = ocr_extract_target_upper(path, self.ocr_engine, fmt, alpha)
            if t:
                new = rename_to_pure_target(path, t)
                return os.path.basename(path), t, new, "ok"
            return os.path.basename(path), None, None, err
        except Exception as e:
            return os.path.basename(path), None, None, str(e)

    def run_batch(self):
        d = self.var_pdf_dir.get()
        files = [os.path.join(d,f) for f in os.listdir(d) if f.lower().endswith(".pdf")]
        if not files:
            self.log("未找到PDF文件")
            self.is_running = False
            return
        total = len(files)
        ok = 0
        ng = 0
        with ThreadPoolExecutor(4) as pool:
            tasks = {pool.submit(self._process_one, f):f for f in files}
            for i,fut in enumerate(as_completed(tasks)):
                if not self.is_running: break
                name = os.path.basename(tasks[fut])
                try:
                    ori, target, newname, status = fut.result()
                    if status == "ok":
                        self.log(f"[{i+1}/{total}] ✅ {ori} → {newname}")
                        ok +=1
                    else:
                        self.log(f"[{i+1}/{total}] ❌ {ori} | {status}")
                        ng +=1
                except:
                    ng +=1
                self.stat_var.set(f"成功：{ok} | 失败：{ng} | 总计：{total}")
        self.log(f"\n处理完成：成功 {ok} 个，失败 {ng} 个")
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.loading_label.config(text="")

    def log(self, msg):
        tm = time.strftime("[%H:%M:%S] ")
        self._log_buffer.append(tm + msg + "\n")
        if len(self._log_buffer)>=10 or "完成" in msg:
            self.log_text.insert(tk.END, "".join(self._log_buffer))
            self._log_buffer = []
            self.log_text.see(tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFOCRRenamerGUI(root)
    root.mainloop()