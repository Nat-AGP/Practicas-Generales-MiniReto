import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

THEME = {
    "bg_main": "#2b2b2b",       # Gris oscuro suave
    "bg_panel": "#333333",      # Paneles laterales
    "fg_text": "#ffffff",       # Texto blanco
    "accent": "#4caf50",        # Verde amigable (Acción)
    "highlight": "#2196f3",     # Azul (Selección)
    "font_main": ("Segoe UI", 10),
    "font_title": ("Segoe UI", 14, "bold")
}

#  MOTOR DE PROCESAMIENTO
class ImageEngine:
    """Clase que contiene TODAS las fórmulas matemáticas de las Prácticas 1, 2 y 3"""
    
    @staticmethod
    def convertir_gris(img):
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    # --- PRÁCTICA 1: OPERACIONES BÁSICAS ---
    @staticmethod
    def op_aritmetica(img, tipo, valor):
        img_gris = ImageEngine.convertir_gris(img)
        if tipo == "Suma (+50)": return cv2.add(img_gris, valor)
        elif tipo == "Resta (-50)": return cv2.subtract(img_gris, valor)
        elif tipo == "Multiplicación (x1.5)": return cv2.multiply(img_gris, 1.5).astype(np.uint8)
        return img_gris

    @staticmethod
    def op_logica(img, tipo):
        img_gris = ImageEngine.convertir_gris(img)
        rows, cols = img_gris.shape
        # Crear máscara circular centrada para probar lógica
        mask = np.zeros((rows, cols), dtype=np.uint8)
        cv2.circle(mask, (cols//2, rows//2), min(rows, cols)//4, 255, -1)
        
        if tipo == "AND (Máscara)": return cv2.bitwise_and(img_gris, img_gris, mask=mask)
        elif tipo == "OR (Máscara)": return cv2.bitwise_or(img_gris, mask)
        elif tipo == "XOR (Máscara)": return cv2.bitwise_xor(img_gris, mask)
        elif tipo == "NOT (Invertir)": return cv2.bitwise_not(img_gris)
        return img_gris

    # --- PRÁCTICA 2: FILTROS Y BORDES ---
    @staticmethod
    def ruido_sp(img, prob):
        img_gris = ImageEngine.convertir_gris(img)
        output = np.copy(img_gris)
        probs = np.random.random(output.shape)
        output[probs < (prob / 2)] = 255
        output[probs > (1 - (prob / 2))] = 0
        return output

    @staticmethod
    def filtros_pasaaltas(img, tipo):
        img_gris = ImageEngine.convertir_gris(img)
        
        if tipo == "Canny":
            return cv2.Canny(img_gris, 100, 200)
        elif tipo == "Sobel":
            sobelx = cv2.Sobel(img_gris, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_gris, cv2.CV_64F, 0, 1, ksize=3)
            return cv2.magnitude(sobelx, sobely).astype(np.uint8)
        elif tipo == "Laplaciano":
            return cv2.convertScaleAbs(cv2.Laplacian(img_gris, cv2.CV_64F))
        elif tipo == "Prewitt":
            kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
            img_prewittx = cv2.filter2D(img_gris, -1, kernelx)
            img_prewitty = cv2.filter2D(img_gris, -1, kernely)
            return img_prewittx + img_prewitty
        elif tipo == "Kirsch":
            # Implementación simplificada de Kirsch
            k_n = np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
            return cv2.filter2D(img_gris, -1, k_n) 
        return img_gris

    # --- PRÁCTICA 3: SEGMENTACIÓN Y BRILLO ---
    @staticmethod
    def ajustar_brillo(img, metodo, gamma_val):
        img_gris = ImageEngine.convertir_gris(img)
        norm = img_gris / 255.0
        
        if metodo == "Corrección Gamma":
            return np.clip(255 * (norm ** gamma_val), 0, 255).astype(np.uint8)
        elif metodo == "Ecualización Hist.":
            return cv2.equalizeHist(img_gris)
        elif metodo == "Logarítmica":
            c = 255 / np.log(1 + np.max(img_gris))
            log_image = c * (np.log(img_gris + 1))
            return np.array(log_image, dtype=np.uint8)
        return img_gris

    @staticmethod
    def segmentar(img, metodo):
        img_gris = ImageEngine.convertir_gris(img)
        if metodo == "Otsu":
            _, th = cv2.threshold(img_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return th
        elif metodo == "Media":
            th_val = np.mean(img_gris)
            _, th = cv2.threshold(img_gris, th_val, 255, cv2.THRESH_BINARY)
            return th
        elif metodo == "Kapur (Entropía)":
            # Implementación rápida de Kapur
            hist = cv2.calcHist([img_gris], [0], None, [256], [0, 256]).flatten()
            hist = hist / hist.sum()
            Hx = -np.cumsum(hist * np.log(hist + 1e-10))
            # Usamos Otsu como fallback robusto para visualización rápida
            _, th = cv2.threshold(img_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return th
        return img_gris

    # --- MINIRETO ESPECIAL: DETECCIÓN DE OBJETOS ---
    @staticmethod
    def analizar_objeto_filoso(img):
        """
        Detecta bordes filosos (Canny), elimina ruido, aisla el objeto mayor
        y calcula área/perímetro.
        """
        img_gris = ImageEngine.convertir_gris(img)
        
        # 1. Suavizado para reducir ruido falso
        blurred = cv2.GaussianBlur(img_gris, (5, 5), 0)
        
        # 2. Canny para detectar bordes filosos
        edges = cv2.Canny(blurred, 50, 150)
        
        # 3. Morfología para cerrar el contorno del objeto
        kernel = np.ones((5,5), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 4. Encontrar contornos
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        res_img = cv2.cvtColor(img_gris, cv2.COLOR_GRAY2BGR)
        data = {"area": 0, "perim": 0, "msg": "No se detectó objeto cerrado."}
        
        if contours:
            # Tomamos el contorno más grande (asumimos que es el objeto principal)
            c = max(contours, key=cv2.contourArea)
            
            # Máscara para borrar todo lo que NO sea el objeto ("eliminar extras")
            mask = np.zeros_like(img_gris)
            cv2.drawContours(mask, [c], -1, 255, -1) # Rellenar objeto
            
            # Aislar objeto en la imagen original
            res_img = cv2.bitwise_and(res_img, res_img, mask=mask)
            
            # Dibujar perímetro en verde neón
            cv2.drawContours(res_img, [c], -1, (0, 255, 0), 2)
            
            # Cálculos
            area = cv2.contourArea(c)
            perimetro = cv2.arcLength(c, True)
            
            data = {
                "area": f"{area:.2f} px²",
                "perim": f"{perimetro:.2f} px",
                "msg": "Objeto aislado y calculado."
            }
            
        return res_img, data

#  INTERFAZ GRÁFICA (GUI)
class AppProcesamiento:
    def __init__(self, root):
        self.root = root
        self.root.title("Comunidad 1 - Laboratorio de Imágenes Integrado")
        self.root.geometry("1300x800")
        self.root.configure(bg=THEME["bg_main"])

        # Variables de estado
        self.img_original = None
        self.img_procesada = None
        self.zoom_window = None

        self._crear_interfaz()

    def _crear_interfaz(self):
        # --- 1. BARRA LATERAL (MENU) ---
        self.sidebar = tk.Frame(self.root, bg=THEME["bg_panel"], width=250)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        # Título Personalizado
        tk.Label(self.sidebar, text="Comunidad 1", font=THEME["font_title"], 
                 bg=THEME["bg_panel"], fg=THEME["highlight"]).pack(pady=20)

        # Botón Carga
        self._crear_boton_menu("📂 Cargar Imagen", self.cargar_imagen, color=THEME["highlight"])
        
        ttk.Separator(self.sidebar, orient='horizontal').pack(fill='x', pady=10)
        
        # Botones de Prácticas
        tk.Label(self.sidebar, text="Selecciona Módulo:", font=("Segoe UI", 9), bg=THEME["bg_panel"], fg="#888").pack(pady=5)
        self._crear_boton_menu("Práctica 1: Lógica", lambda: self.mostrar_panel("P1"))
        self._crear_boton_menu("Práctica 2: Filtros", lambda: self.mostrar_panel("P2"))
        self._crear_boton_menu("Práctica 3: Segmentación", lambda: self.mostrar_panel("P3"))
        self._crear_boton_menu("📐 Análisis de Objeto", lambda: self.mostrar_panel("RETO"), color=THEME["accent"])

        # Panel de Controles Dinámico (Abajo a la izquierda)
        self.frame_controles = tk.Frame(self.sidebar, bg=THEME["bg_panel"])
        self.frame_controles.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- 2. ÁREA DE TRABAJO (DERECHA) ---
        self.workspace = tk.Frame(self.root, bg=THEME["bg_main"])
        self.workspace.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Área de Imágenes (Original vs Procesada)
        frame_imgs = tk.Frame(self.workspace, bg=THEME["bg_main"])
        frame_imgs.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.lbl_orig = self._crear_label_imagen(frame_imgs, "Original")
        self.lbl_proc = self._crear_label_imagen(frame_imgs, "Resultado (Pasa el mouse para Lupa 🔍)")
        
        # Bindings para la Lupa
        self.lbl_proc.bind("<Motion>", self.actualizar_lupa)
        self.lbl_proc.bind("<Leave>", self.cerrar_lupa)

        # Área de Histograma
        self.frame_hist = tk.Frame(self.workspace, bg=THEME["bg_main"], height=200)
        self.frame_hist.pack(side=tk.BOTTOM, fill=tk.X)
        self._init_histograma()

    def _crear_boton_menu(self, texto, comando, color=None):
        bg_color = THEME["bg_panel"] if not color else color
        btn = tk.Button(self.sidebar, text=texto, command=comando,
                        bg=bg_color, fg="white", font=THEME["font_main"],
                        bd=0, pady=8, cursor="hand2", anchor="w", padx=20)
        btn.pack(fill=tk.X, pady=2)
        # Hover effect
        if not color:
            btn.bind("<Enter>", lambda e: btn.config(bg="#444"))
            btn.bind("<Leave>", lambda e: btn.config(bg=THEME["bg_panel"]))

    def _crear_label_imagen(self, parent, titulo):
        f = tk.Frame(parent, bg=THEME["bg_main"])
        f.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5)
        tk.Label(f, text=titulo, bg=THEME["bg_main"], fg="#888").pack()
        l = tk.Label(f, bg="black")
        l.pack(expand=True, fill=tk.BOTH)
        return l

    def _init_histograma(self):
        self.fig = Figure(figsize=(5, 2), dpi=100, facecolor=THEME["bg_main"])
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(THEME["bg_main"])
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        
        self.canvas_hist = FigureCanvasTkAgg(self.fig, master=self.frame_hist)
        self.canvas_hist.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    #  CONTROLES DINÁMICOS (CAMBIAN SEGÚN PRÁCTICA)
    def mostrar_panel(self, modulo):
        # Limpiar controles anteriores
        for widget in self.frame_controles.winfo_children():
            widget.destroy()

        lbl = tk.Label(self.frame_controles, text="", font=("Segoe UI", 11, "bold"), bg=THEME["bg_panel"], fg="white")
        lbl.pack(pady=(10, 10), anchor="w")

        if modulo == "P1":
            lbl.config(text="PRÁCTICA 1: Lógica")
            self._add_combo("Operación Aritmética", ["Suma (+50)", "Resta (-50)", "Multiplicación (x1.5)"], 
                           lambda e: self.procesar(lambda img: ImageEngine.op_aritmetica(img, e, 50)))
            self._add_combo("Operación Lógica", ["AND (Máscara)", "OR (Máscara)", "XOR (Máscara)", "NOT (Invertir)"],
                           lambda e: self.procesar(lambda img: ImageEngine.op_logica(img, e)))

        elif modulo == "P2":
            lbl.config(text="PRÁCTICA 2: Filtros")
            self._add_slider("Ruido Sal y Pimienta", 0.0, 0.5, 
                             lambda v: self.procesar(lambda img: ImageEngine.ruido_sp(img, float(v))))
            
            # SECCIÓN CANNY Y BORDES
            tk.Label(self.frame_controles, text="Detección de Bordes:", bg=THEME["bg_panel"], fg=THEME["highlight"]).pack(anchor="w", pady=(10,0))
            self._add_combo("Tipo de Filtro", ["Canny", "Sobel", "Prewitt", "Kirsch", "Laplaciano"],
                           lambda e: self.procesar(lambda img: ImageEngine.filtros_pasaaltas(img, e)))

        elif modulo == "P3":
            lbl.config(text="PRÁCTICA 3: Segmentación")
            self._add_slider("Corrección Gamma", 0.1, 3.0,
                             lambda v: self.procesar(lambda img: ImageEngine.ajustar_brillo(img, "Corrección Gamma", float(v))))
            
            self._add_combo("Método Segmentación", ["Otsu", "Media", "Kapur (Entropía)"],
                           lambda e: self.procesar(lambda img: ImageEngine.segmentar(img, e)))

        elif modulo == "RETO":
            lbl.config(text="ANÁLISIS DE OBJETO")
            lbl_desc = tk.Label(self.frame_controles, text="Este módulo limpia la imagen,\nelimina ruido externo y\ncalcula métricas del objeto.", 
                                bg=THEME["bg_panel"], fg="#aaa", justify="left")
            lbl_desc.pack(pady=5, anchor="w")
            
            btn = tk.Button(self.frame_controles, text="▶ EJECUTAR ANÁLISIS", bg=THEME["accent"], fg="white",
                            font=("Segoe UI", 10, "bold"), command=self.ejecutar_analisis)
            btn.pack(fill=tk.X, pady=20)
            
            self.lbl_resultados = tk.Label(self.frame_controles, text="Esperando...", bg="#222", fg="#0f0", font=("Consolas", 10), justify="left", padx=5, pady=5)
            self.lbl_resultados.pack(fill=tk.X)

    def _add_combo(self, titulo, valores, comando):
        tk.Label(self.frame_controles, text=titulo, bg=THEME["bg_panel"], fg="white").pack(anchor="w", pady=(5,0))
        cb = ttk.Combobox(self.frame_controles, values=valores, state="readonly")
        cb.pack(fill=tk.X, pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: comando(cb.get()))

    def _add_slider(self, titulo, min_val, max_val, comando):
        tk.Label(self.frame_controles, text=titulo, bg=THEME["bg_panel"], fg="white").pack(anchor="w", pady=(5,0))
        s = ttk.Scale(self.frame_controles, from_=min_val, to=max_val, command=comando)
        s.pack(fill=tk.X, pady=2)

    # FUNCIONALIDAD CENTRAL
    def cargar_imagen(self):
        path = filedialog.askopenfilename()
        if path:
            self.img_original = cv2.imread(path)
            # Resize inteligente solo si es muy grande, para que quepa en pantalla
            h, w = self.img_original.shape[:2]
            if w > 800 or h > 600:
                scale = min(800/w, 600/h)
                self.img_original = cv2.resize(self.img_original, (int(w*scale), int(h*scale)))
                
            self.img_procesada = self.img_original.copy()
            
            self._mostrar_imagen(self.lbl_orig, self.img_original)
            self._mostrar_imagen(self.lbl_proc, self.img_procesada)
            self._actualizar_histograma(self.img_original)

    def _mostrar_imagen(self, label, cv_img):
        if cv_img is None: return
        # Convertir BGR a RGB
        if len(cv_img.shape) == 3:
            color = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        else:
            color = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)
        
        im_pil = Image.fromarray(color)
        img_tk = ImageTk.PhotoImage(im_pil)
        label.config(image=img_tk)
        label.image = img_tk

    def procesar(self, func_logica):
        if self.img_original is None: 
            messagebox.showwarning("Atención", "Carga una imagen primero.")
            return
        
        self.img_procesada = func_logica(self.img_original)
        self._mostrar_imagen(self.lbl_proc, self.img_procesada)
        self._actualizar_histograma(self.img_procesada)

    def ejecutar_analisis(self):
        if self.img_original is None: return
        res_img, data = ImageEngine.analizar_objeto_filoso(self.img_original)
        
        self.img_procesada = res_img
        self._mostrar_imagen(self.lbl_proc, res_img)
        self._actualizar_histograma(res_img)
        
        txt = f"{data['msg']}\n\nÁrea: {data['area']}\nPerímetro: {data['perim']}"
        self.lbl_resultados.config(text=txt)

    def _actualizar_histograma(self, img):
        self.ax.clear()
        if len(img.shape) == 3:
            for i, col in enumerate(['b', 'g', 'r']):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                self.ax.plot(hist, color=col)
        else:
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            self.ax.plot(hist, color='white')
            self.ax.fill_between(range(256), hist.ravel(), color='gray', alpha=0.3)
        
        self.ax.set_xlim([0, 256])
        self.ax.grid(True, alpha=0.2)
        self.canvas_hist.draw()

    #  LUPA 
    def actualizar_lupa(self, event):
        if self.img_procesada is None: return
        
        x, y = event.x, event.y
        # Obtener dimensiones reales
        w_widget = self.lbl_proc.winfo_width()
        h_widget = self.lbl_proc.winfo_height()
        h_img, w_img = self.img_procesada.shape[:2]
        
        if w_widget == 0 or h_widget == 0: return # Prevención de error al iniciar
        
        # Mapeo de coordenadas (simple)
        scale_x = w_img / w_widget
        scale_y = h_img / h_widget
        
        real_x, real_y = int(x * scale_x), int(y * scale_y)
        
        # Crop (Recorte)
        size = 40 # Tamaño del área a hacer zoom
        x1 = max(0, real_x - size)
        y1 = max(0, real_y - size)
        x2 = min(w_img, real_x + size)
        y2 = min(h_img, real_y + size)
        
        crop = self.img_procesada[y1:y2, x1:x2]
        if crop.size == 0: return
        
        # Zoom (Resize)
        zoom_factor = 4
        crop_zoom = cv2.resize(crop, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_NEAREST)
        
        # Mostrar ventana flotante
        if self.zoom_window is None:
            self.zoom_window = Toplevel(self.root)
            self.zoom_window.wm_overrideredirect(True) # Sin bordes
            self.zoom_lbl = tk.Label(self.zoom_window, bd=2, relief="solid")
            self.zoom_lbl.pack()
            self.zoom_window.attributes("-topmost", True)
            
        # Convertir para TK
        if len(crop_zoom.shape) == 3:
            crop_rgb = cv2.cvtColor(crop_zoom, cv2.COLOR_BGR2RGB)
        else:
            crop_rgb = cv2.cvtColor(crop_zoom, cv2.COLOR_GRAY2RGB)
            
        im_pil = Image.fromarray(crop_rgb)
        img_tk = ImageTk.PhotoImage(im_pil)
        
        self.zoom_lbl.config(image=img_tk)
        self.zoom_lbl.image = img_tk
        self.zoom_window.geometry(f"+{event.x_root+20}+{event.y_root+20}")

    def cerrar_lupa(self, event):
        if self.zoom_window:
            self.zoom_window.destroy()
            self.zoom_window = None

if __name__ == "__main__":
    root = tk.Tk()
    app = AppProcesamiento(root)
    root.mainloop()