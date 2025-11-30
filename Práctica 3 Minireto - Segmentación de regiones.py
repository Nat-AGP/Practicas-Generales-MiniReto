import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks

class Practica3App:
    def __init__(self, root):
        self.root = root
        self.root.title("Práctica 3: Segmentación y Ajuste de Brillo")
        self.root.geometry("1200x800")
        
        # Estilos modernos (Tema oscuro/profesional)
        style = ttk.Style()
        style.theme_use('clam')
        
        # Variables de estado
        self.original_image = None
        self.gray_image = None
        self.processed_image = None
        self.current_hist_data = None
        
        # --- Layout Principal ---
        # Panel izquierdo (Controles)
        self.panel_left = ttk.Frame(root, padding="10")
        self.panel_left.pack(side=tk.LEFT, fill=tk.Y)
        
        # Panel derecho (Visualización)
        self.panel_right = ttk.Frame(root, padding="10")
        self.panel_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self._init_controls()
        self._init_display()

    def _init_controls(self):
        # Sección de Carga
        lbl_title = ttk.Label(self.panel_left, text="Controles", font=("Helvetica", 16, "bold"))
        lbl_title.pack(pady=(0, 20))
        
        btn_load = ttk.Button(self.panel_left, text="Cargar Imagen", command=self.load_image)
        btn_load.pack(fill=tk.X, pady=5)
        
        # Sección: Preprocesamiento (Ruido) 
        ttk.Label(self.panel_left, text="--- Ruido / Preprocesamiento ---").pack(pady=(15, 5))
        self.noise_var = tk.StringVar(value="Ninguno")
        cb_noise = ttk.Combobox(self.panel_left, textvariable=self.noise_var, state="readonly")
        cb_noise['values'] = ("Ninguno", "Sal y Pimienta", "Gaussiano", "Suavizado (Blur)")
        cb_noise.pack(fill=tk.X)
        cb_noise.bind("<<ComboboxSelected>>", self.apply_transformations)

        # Sección: Ajuste de Brillo [cite: 38-46]
        ttk.Label(self.panel_left, text="--- Ajuste de Brillo ---").pack(pady=(15, 5))
        self.bright_method_var = tk.StringVar(value="Original")
        cb_bright = ttk.Combobox(self.panel_left, textvariable=self.bright_method_var, state="readonly")
        cb_bright['values'] = (
            "Original", 
            "Desplazamiento (+Brillo)", 
            "Desplazamiento (-Brillo)",
            "Expansión Histograma", 
            "Contracción Histograma",
            "Corrección Gamma", 
            "Ec. Uniforme", 
            "Ec. Exponencial", 
            "Ec. Rayleigh", 
            "Ec. Hipercúbica", 
            "Ec. Log. Hiperbólica"
        )
        cb_bright.pack(fill=tk.X)
        cb_bright.bind("<<ComboboxSelected>>", self.apply_transformations)

        # Sliders auxiliares (Gamma / Desplazamiento)
        self.slider_val = tk.DoubleVar(value=1.0)
        self.slider = ttk.Scale(self.panel_left, from_=0.1, to=3.0, variable=self.slider_val, orient=tk.HORIZONTAL)
        self.slider.pack(fill=tk.X, pady=5)
        self.slider.bind("<ButtonRelease-1>", self.apply_transformations)
        ttk.Label(self.panel_left, text="Factor (Gamma/Shift/Etc)").pack()

        # Sección: Segmentación (Umbralado) [cite: 31-37]
        ttk.Label(self.panel_left, text="--- Segmentación ---").pack(pady=(15, 5))
        self.seg_method_var = tk.StringVar(value="Ninguna")
        cb_seg = ttk.Combobox(self.panel_left, textvariable=self.seg_method_var, state="readonly")
        cb_seg['values'] = (
            "Ninguna",
            "Otsu",
            "Entropía de Kapur",
            "Mínimo Histograma",
            "Media",
            "Umbral Banda (80-150)"
        )
        cb_seg.pack(fill=tk.X)
        cb_seg.bind("<<ComboboxSelected>>", self.apply_transformations)

    def _init_display(self):
        # Figura de Matplotlib integrada en Tkinter
        self.fig = plt.figure(figsize=(8, 8), dpi=100)
        self.ax_img = self.fig.add_subplot(2, 1, 1)
        self.ax_hist = self.fig.add_subplot(2, 1, 2)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.panel_right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Imagenes", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            # Carga en escala de grises como sugiere la práctica 
            # Pero leemos en color y convertimos para mejor manejo interno si fuera necesario
            img_bgr = cv2.imread(path)
            self.original_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            self.gray_image = self.original_image.copy()
            self.apply_transformations()

    # --- LÓGICA DE PROCESAMIENTO ---
    
    def apply_transformations(self, event=None):
        if self.original_image is None:
            return

        # 1. Base: Imagen Original en Grises
        img = self.original_image.copy()

        # 2. Aplicar Ruido 
        noise_type = self.noise_var.get()
        if noise_type == "Sal y Pimienta":
            row, col = img.shape
            s_vs_p = 0.5
            amount = 0.04
            out = np.copy(img)
            # Sal
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
            out[tuple(coords)] = 255
            # Pimienta
            num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
            out[tuple(coords)] = 0
            img = out
        elif noise_type == "Gaussiano":
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, img.shape).reshape(img.shape)
            noisy = img + gauss * 50 # Factor para que se note
            img = np.clip(noisy, 0, 255).astype(np.uint8)
        elif noise_type == "Suavizado (Blur)":
            img = cv2.GaussianBlur(img, (5, 5), 0)

        # 3. Ajuste de Brillo y Ecualización [cite: 38-46]
        bright_method = self.bright_method_var.get()
        val = self.slider_val.get() # Valor del slider

        norm_img = img / 255.0 # Normalizado 0-1 para formulas matemáticas
        
        if bright_method == "Ec. Uniforme": # [cite: 38]
            img = cv2.equalizeHist(img)
        elif bright_method == "Ec. Exponencial": # [cite: 39]
            # Formula aproximada basada en PDF: 255 * (1 - exp(-img/255)) [cite: 392]
            # Nota: El PDF tiene un typo en formula, usamos la logica standard
            res = 255 * (1 - np.exp(-norm_img)) # Ajuste simple
            img = np.clip(res / np.max(res) * 255, 0, 255).astype(np.uint8)
        elif bright_method == "Ec. Rayleigh": # [cite: 40]
            # PDF: 255 * sqrt(img/255) [cite: 398]
            # Nota: Rayleigh real usa distribución, el PDF usa una transformación raíz simple
            res = 255 * np.sqrt(norm_img)
            img = np.clip(res, 0, 255).astype(np.uint8)
        elif bright_method == "Ec. Hipercúbica": # [cite: 41]
            # PDF: 255 * (img/255)^4 [cite: 404]
            res = 255 * (norm_img ** 4)
            img = np.clip(res, 0, 255).astype(np.uint8)
        elif bright_method == "Ec. Log. Hiperbólica": # [cite: 42]
            # PDF: 255 * log(1+img)/log(1+255) [cite: 411]
            # Nota: Aplicar sobre valores 0-255 o normalizados
            res = 255 * (np.log1p(img.astype(float)) / np.log1p(255.0))
            img = np.clip(res, 0, 255).astype(np.uint8)
        elif bright_method == "Corrección Gamma": # [cite: 44, 208]
            # gamma < 1 aclara, > 1 oscurece
            gamma = val
            res = 255 * np.power(norm_img, gamma)
            img = np.clip(res, 0, 255).astype(np.uint8)
        elif bright_method == "Desplazamiento (+Brillo)": # [cite: 45]
            shift = int(val * 50)
            img = cv2.add(img, shift) # cv2.add evita overflow (clipping)
        elif bright_method == "Desplazamiento (-Brillo)": # [cite: 45]
            shift = int(val * 50)
            img = cv2.subtract(img, shift)
        elif bright_method == "Expansión Histograma": # [cite: 46]
            # Contrast stretching
            min_val, max_val = np.min(img), np.max(img)
            img = ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        self.processed_image = img # Guardamos estado pre-segmentación para histograma

        # 4. Segmentación [cite: 31-37]
        seg_method = self.seg_method_var.get()
        final_show = img
        threshold_val = 0

        if seg_method == "Otsu": # [cite: 31, 236]
            threshold_val, final_show = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif seg_method == "Media": # [cite: 36, 315]
            threshold_val = np.mean(img)
            final_show = (img >= threshold_val).astype(np.uint8) * 255
            
        elif seg_method == "Mínimo Histograma": # [cite: 34, 298]
            hist, bins = np.histogram(img.flatten(), 256, [0, 256])
            # Encontrar picos con distancia mínima (ajustable)
            peaks, _ = find_peaks(hist, distance=20)
            if len(peaks) >= 2:
                # Buscar mínimo entre el primer y segundo pico
                p1, p2 = peaks[0], peaks[1]
                # Slice del histograma entre picos
                section = hist[p1:p2]
                relative_min = np.argmin(section)
                threshold_val = p1 + relative_min
            else:
                threshold_val = 127 # Fallback
            final_show = (img > threshold_val).astype(np.uint8) * 255

        elif seg_method == "Entropía de Kapur": # 
            # Implementación optimizada vectorizada
            hist, _ = np.histogram(img.flatten(), 256, [0, 256])
            total_pixels = img.size
            prob = hist / total_pixels
            
            # Tablas de probabilidad acumulada (P0, P1) y entropía acumulada (H0, H1)
            # Nota: Esto es pesado para Python puro, simplificamos con un loop robusto
            max_entropy = -1.0
            threshold_val = 0
            
            # Pre-cálculo para evitar log(0)
            prob_nz = prob.copy()
            prob_nz[prob_nz == 0] = 1e-10
            
            # Vector de Entropía Acumulada
            H = -np.cumsum(prob * np.log(prob_nz))
            P = np.cumsum(prob)
            
            # Iterar t para buscar max entropía
            for t in range(1, 256):
                # Clase 1 (Fondo): 0 a t
                w0 = P[t]
                if w0 == 0: continue
                # Clase 2 (Objeto): t+1 a 255
                w1 = 1.0 - w0
                if w1 <= 0: continue

                # Entropía normalizada
                # Ha = sum(pi * ln(pi)) / w0 + ln(w0) -> formula simplificada Kapur
                # Usando formula PDF[cite: 277]: H_total = H_obj + H_bg
                
                # Recalcular manual para coincidir con PDF logic [cite: 268-273]
                h0 = -np.sum((prob[:t+1]/w0) * np.log(prob_nz[:t+1]/w0))
                h1 = -np.sum((prob[t+1:]/w1) * np.log(prob_nz[t+1:]/w1))
                
                total_entropy = h0 + h1
                
                if total_entropy > max_entropy:
                    max_entropy = total_entropy
                    threshold_val = t
            
            final_show = (img > threshold_val).astype(np.uint8) * 255

        elif seg_method == "Umbral Banda (80-150)": # [cite: 37, 341]
            t1, t2 = 80, 150
            # Crear máscara negra
            final_show = np.zeros_like(img)
            # Píxeles dentro del rango a blanco
            mask = (img >= t1) & (img <= t2)
            final_show[mask] = 255
            threshold_val = f"{t1}-{t2}"

        # --- Actualizar GUI ---
        self._update_plot(img, final_show, seg_method, threshold_val)

    def _update_plot(self, img_pre_seg, img_final, seg_name, thresh_val):
        self.ax_img.clear()
        self.ax_hist.clear()
        
        # Imagen Principal
        self.ax_img.imshow(img_final, cmap='gray')
        title = f"Resultado: {seg_name}"
        if thresh_val != 0:
            title += f" (Umbral: {thresh_val})"
        self.ax_img.set_title(title)
        self.ax_img.axis('off')
        
        # Histograma (De la imagen PRE-segmentación para ver distribución de grises) 
        self.ax_hist.hist(img_pre_seg.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
        self.ax_hist.set_title("Histograma (Imagen ajustada, previo a umbral)")
        self.ax_hist.set_xlim([0, 256])
        
        # Marcar el umbral si es un número único
        if isinstance(thresh_val, (int, float, np.integer)) and thresh_val > 0:
            self.ax_hist.axvline(thresh_val, color='r', linestyle='--', label=f'Umbral: {thresh_val}')
            self.ax_hist.legend()

        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = Practica3App(root)
    root.mainloop()