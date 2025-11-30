import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# ==========================================
# LOGICA DE PROCESAMIENTO DE IMAGENES
# ==========================================

class ImageProcessor:
    """
    Clase encargada de la lógica algorítmica.
    """

    @staticmethod
    def agregar_ruido_sal_pimienta(imagen: np.ndarray, probabilidad: float) -> np.ndarray:
        """
        Agrega ruido de sal y pimienta a una imagen en escala de grises.
        """
        if probabilidad <= 0:
            return imagen.copy()

        output = np.copy(imagen)
        probs = np.random.random(output.shape)
        output[probs < (probabilidad / 2)] = 255
        output[probs > (1 - (probabilidad / 2))] = 0
        return output

    @staticmethod
    def agregar_ruido_gaussiano(imagen: np.ndarray, sigma: int) -> np.ndarray:
        """
        Agrega ruido gaussiano aditivo a una imagen en escala de grises.
        """
        if sigma <= 0:
            return imagen.copy()

        mean = 0
        gauss = np.random.normal(mean, sigma, imagen.shape)
        noisy = imagen.astype(np.float32) + gauss
        noisy_clipped = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy_clipped

    @staticmethod
    def aplicar_filtro_restauracion(imagen: np.ndarray, tipo_filtro: str, ksize: int) -> np.ndarray:
        """
        Aplica filtros de restauración (pasabajas).
        """
        if ksize % 2 == 0:
            ksize += 1

        if tipo_filtro == "Mediana (Sal y Pimienta)":
            return cv2.medianBlur(imagen, ksize)
        elif tipo_filtro == "Gaussiano (Suavizado)":
            return cv2.GaussianBlur(imagen, (ksize, ksize), 0)
        elif tipo_filtro == "Promedio (Blur)":
            return cv2.blur(imagen, (ksize, ksize))
        return imagen

    @staticmethod
    def aplicar_filtro_pasaaltas(imagen: np.ndarray, tipo_filtro: str, param1: int = 100, param2: int = 200) -> np.ndarray:
        """
        Aplica filtros pasaaltas (detección de bordes).
        """
        # Asegurar que la imagen esté en escala de grises
        if len(imagen.shape) == 3:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        if tipo_filtro == "Roberts":
            # Usamos CV_64F para evitar recortar valores negativos en los bordes
            kernel_roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
            kernel_roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            bordes_x = cv2.filter2D(imagen, cv2.CV_64F, kernel_roberts_x)
            bordes_y = cv2.filter2D(imagen, cv2.CV_64F, kernel_roberts_y)
            # Magnitud absoluta y conversión
            bordes = cv2.magnitude(bordes_x, bordes_y)
            return np.clip(bordes, 0, 255).astype(np.uint8)

        elif tipo_filtro == "Sobel":
            sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
            bordes = cv2.magnitude(sobel_x, sobel_y)
            return np.uint8(np.clip(bordes, 0, 255))

        elif tipo_filtro == "Prewitt":
            kernel_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
            kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
            bordes_x = cv2.filter2D(imagen, cv2.CV_64F, kernel_prewitt_x)
            bordes_y = cv2.filter2D(imagen, cv2.CV_64F, kernel_prewitt_y)
            bordes = cv2.addWeighted(np.abs(bordes_x), 0.5, np.abs(bordes_y), 0.5, 0)
            return np.uint8(bordes)

        elif tipo_filtro == "Canny":
            return cv2.Canny(imagen, param1, param2)

        elif tipo_filtro == "Kirsch":
            kirsch_kernels = [
                np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
                np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
                np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
                np.array([[-3, -3, -3], [-3, 0, 5], [5, 5, 5]]),
                np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
                np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
                np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
                np.array([[5, 5, 5], [5, 0, -3], [-3, -3, -3]])
            ]
            # Aplicar filtros y mantener el máximo de respuesta
            bordes = np.max([cv2.filter2D(imagen, cv2.CV_64F, k) for k in kirsch_kernels], axis=0)
            return np.uint8(np.clip(np.abs(bordes), 0, 255))

        elif tipo_filtro == "Laplaciano":
            laplaciano = cv2.Laplacian(imagen, cv2.CV_64F)
            return np.uint8(np.clip(np.abs(laplaciano), 0, 255))

        return imagen


# ==========================================
# INTERFAZ GRÁFICA DE USUARIO (GUI)
# ==========================================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("PDI: Restauración y Filtros Pasaaltas")
        self.root.geometry("1500x800")

        # Variables de estado
        self.original_image_color = None
        self.noisy_image = None
        self.processed_image = None
        
        # Modo de operación
        self.modo_var = tk.StringVar(value="Restauración")

        self._init_ui()

    def _init_ui(self):
        """Configura los widgets de la ventana."""

        # --- Panel Izquierdo (Controles) ---
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # SELECTOR DE MODO
        ttk.Label(control_frame, text="Modo de Operación", font=("Arial", 12, "bold")).pack(pady=10)
        
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(mode_frame, text="Restauración", variable=self.modo_var, 
                        value="Restauración", command=self.cambiar_modo).pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="Filtros Pasaaltas", variable=self.modo_var, 
                        value="Pasaaltas", command=self.cambiar_modo).pack(anchor="w")

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # 1. Carga
        ttk.Label(control_frame, text="1. Cargar Imagen", font=("Arial", 10, "bold")).pack(pady=5)
        self.btn_load = ttk.Button(control_frame, text="Seleccionar Imagen", command=self.cargar_imagen)
        self.btn_load.pack(fill=tk.X, pady=5)

        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)

        # FRAME PARA CONTROLES DE RESTAURACIÓN
        self.frame_restauracion = ttk.Frame(control_frame)
        self.frame_restauracion.pack(fill=tk.BOTH, expand=True)
        self._crear_controles_restauracion(self.frame_restauracion)

        # FRAME PARA CONTROLES DE PASAALTAS
        self.frame_pasaaltas = ttk.Frame(control_frame)
        self._crear_controles_pasaaltas(self.frame_pasaaltas)
        # Ocultar pasaaltas inicialmente
        self.frame_pasaaltas.pack_forget()

        # --- Panel Derecho (Visualización) ---
        display_frame = ttk.Frame(self.root)
        display_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Panel 1: Referencia Original
        self.panel_clean = self._crear_panel_imagen(display_frame, "1. Original")
        # Panel 2: Intermedio
        self.panel_noisy = self._crear_panel_imagen(display_frame, "2. Procesado (Ruido / Grises)")
        # Panel 3: Resultado
        self.panel_resultado = self._crear_panel_imagen(display_frame, "3. Resultado Final")

    def _crear_controles_restauracion(self, parent):
        """Controles para el modo de Restauración"""
        ttk.Label(parent, text="2. Generar Ruido", font=("Arial", 10, "bold")).pack(pady=5)

        self.ruido_var = tk.StringVar(value="Original")

        ttk.Radiobutton(parent, text="Original (Color)", variable=self.ruido_var, 
                        value="Original", command=self.actualizar_imagen).pack(anchor="w")
        ttk.Radiobutton(parent, text="Sal y Pimienta (Grises)", variable=self.ruido_var, 
                        value="SP", command=self.actualizar_imagen).pack(anchor="w")
        ttk.Radiobutton(parent, text="Gaussiano (Grises)", variable=self.ruido_var, 
                        value="Gauss", command=self.actualizar_imagen).pack(anchor="w")

        ttk.Label(parent, text="Intensidad S&P:").pack(pady=(10, 0))
        self.slider_sp = ttk.Scale(parent, from_=0.01, to=0.3, 
                                   command=lambda x: self.actualizar_imagen())
        self.slider_sp.set(0.05)
        self.slider_sp.pack(fill=tk.X)

        ttk.Label(parent, text="Sigma Gaussiano:").pack(pady=(10, 0))
        self.slider_gauss = ttk.Scale(parent, from_=0, to=100, 
                                      command=lambda x: self.actualizar_imagen())
        self.slider_gauss.set(20)
        self.slider_gauss.pack(fill=tk.X)

        ttk.Separator(parent, orient='horizontal').pack(fill='x', pady=10)

        ttk.Label(parent, text="3. Aplicar Filtro Restauración", font=("Arial", 10, "bold")).pack(pady=5)

        self.filtro_rest_var = tk.StringVar()
        self.combo_filtro_rest = ttk.Combobox(parent, textvariable=self.filtro_rest_var, state="readonly")
        self.combo_filtro_rest['values'] = ("Ninguno", "Mediana (Sal y Pimienta)", 
                                            "Gaussiano (Suavizado)", "Promedio (Blur)")
        self.combo_filtro_rest.current(0)
        self.combo_filtro_rest.bind("<<ComboboxSelected>>", lambda x: self.actualizar_imagen())
        self.combo_filtro_rest.pack(fill=tk.X, pady=5)

        ttk.Label(parent, text="Tamaño Kernel (k):").pack(pady=(5, 0))
        self.slider_kernel = ttk.Scale(parent, from_=3, to=15, 
                                      command=lambda x: self.actualizar_imagen())
        self.slider_kernel.set(3)
        self.slider_kernel.pack(fill=tk.X)

        self.lbl_kernel_val = ttk.Label(parent, text="k = 3")
        self.lbl_kernel_val.pack()

    def _crear_controles_pasaaltas(self, parent):
        """Controles para el modo de Filtros Pasaaltas"""
        ttk.Label(parent, text="2. Filtros de Detección de Bordes", 
                  font=("Arial", 10, "bold")).pack(pady=5)

        self.filtro_pa_var = tk.StringVar()
        self.combo_filtro_pa = ttk.Combobox(parent, textvariable=self.filtro_pa_var, state="readonly")
        self.combo_filtro_pa['values'] = ("Ninguno", "Roberts", "Sobel", "Prewitt", 
                                          "Canny", "Kirsch", "Laplaciano")
        self.combo_filtro_pa.current(0)
        self.combo_filtro_pa.bind("<<ComboboxSelected>>", lambda x: self.actualizar_imagen())
        self.combo_filtro_pa.pack(fill=tk.X, pady=5)

        ttk.Label(parent, text="Parámetros Canny:", font=("Arial", 9, "bold")).pack(pady=(10, 5))
        
        ttk.Label(parent, text="Umbral Inferior:").pack()
        self.slider_canny1 = ttk.Scale(parent, from_=0, to=255, 
                                      command=lambda x: self.actualizar_imagen())
        self.slider_canny1.set(100)
        self.slider_canny1.pack(fill=tk.X)
        self.lbl_canny1 = ttk.Label(parent, text="100")
        self.lbl_canny1.pack()

        ttk.Label(parent, text="Umbral Superior:").pack(pady=(5, 0))
        self.slider_canny2 = ttk.Scale(parent, from_=0, to=255, 
                                      command=lambda x: self.actualizar_imagen())
        self.slider_canny2.set(200)
        self.slider_canny2.pack(fill=tk.X)
        self.lbl_canny2 = ttk.Label(parent, text="200")
        self.lbl_canny2.pack()

    def _crear_panel_imagen(self, parent, titulo):
        frame = ttk.Frame(parent, padding=5)
        frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        ttk.Label(frame, text=titulo, font=("Arial", 11, "bold")).pack()
        lbl_img = ttk.Label(frame)
        lbl_img.pack(expand=True)
        return lbl_img

    # ==========================================
    # LÓGICA DEL SISTEMA
    # ==========================================

    def cambiar_modo(self):
        """Cambia entre los dos modos de operación"""
        if self.modo_var.get() == "Restauración":
            self.frame_pasaaltas.pack_forget()
            self.frame_restauracion.pack(fill=tk.BOTH, expand=True)
        else:
            self.frame_restauracion.pack_forget()
            self.frame_pasaaltas.pack(fill=tk.BOTH, expand=True)
        
        self.actualizar_imagen()

    def cargar_imagen(self):
        """Carga la imagen siempre en COLOR (BGR) inicialmente."""
        ruta = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.png *.jpeg *.bmp")])
        if not ruta: return

        try:
            img = cv2.imread(ruta)
            if img is None:
                raise ValueError("No se pudo leer el archivo de imagen.")

            h, w = img.shape[:2]
            max_w = 450
            if w > max_w:
                scale = max_w / w
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            self.original_image_color = img
            self.actualizar_imagen()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def actualizar_imagen(self):
        """Controla el flujo según el modo seleccionado"""
        if self.original_image_color is None: 
            return

        modo = self.modo_var.get()

        if modo == "Restauración":
            self._procesar_restauracion()
        else:
            self._procesar_pasaaltas()

        # Mostrar en los 3 paneles
        self._mostrar_cv2_en_tkinter(self.original_image_color, self.panel_clean)
        self._mostrar_cv2_en_tkinter(self.noisy_image, self.panel_noisy)
        self._mostrar_cv2_en_tkinter(self.processed_image, self.panel_resultado)

    def _procesar_restauracion(self):
        """Procesamiento para modo Restauración"""
        opcion_ruido = self.ruido_var.get()

        if opcion_ruido == "Original":
            img_to_process = self.original_image_color.copy()
        else:
            img_to_process = cv2.cvtColor(self.original_image_color, cv2.COLOR_BGR2GRAY)

            if opcion_ruido == "SP":
                amount = self.slider_sp.get()
                img_to_process = ImageProcessor.agregar_ruido_sal_pimienta(img_to_process, amount)
            elif opcion_ruido == "Gauss":
                sigma = self.slider_gauss.get()
                img_to_process = ImageProcessor.agregar_ruido_gaussiano(img_to_process, sigma)

        self.noisy_image = img_to_process

        tipo_filtro = self.filtro_rest_var.get()
        k_val = int(self.slider_kernel.get())
        if k_val % 2 == 0: k_val += 1
        self.lbl_kernel_val.config(text=f"k = {k_val}")

        if tipo_filtro != "Ninguno":
            self.processed_image = ImageProcessor.aplicar_filtro_restauracion(
                self.noisy_image, tipo_filtro, k_val)
        else:
            self.processed_image = self.noisy_image

    def _procesar_pasaaltas(self):
        """Procesamiento para modo Filtros Pasaaltas"""
        # Convertir a grises para procesamiento
        img_gray = cv2.cvtColor(self.original_image_color, cv2.COLOR_BGR2GRAY)
        self.noisy_image = img_gray

        tipo_filtro = self.filtro_pa_var.get()
        
        if tipo_filtro != "Ninguno":
            # Actualizar labels de Canny
            canny1 = int(self.slider_canny1.get())
            canny2 = int(self.slider_canny2.get())
            self.lbl_canny1.config(text=str(canny1))
            self.lbl_canny2.config(text=str(canny2))
            
            self.processed_image = ImageProcessor.aplicar_filtro_pasaaltas(
                img_gray, tipo_filtro, canny1, canny2)
        else:
            self.processed_image = img_gray

    def _mostrar_cv2_en_tkinter(self, cv_img, label_widget):
        """Convierte la imagen OpenCV (BGR o Grises) a formato Tkinter."""
        if cv_img is None: return

        if len(cv_img.shape) == 3:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB)

        pil_img = Image.fromarray(rgb_img)
        tk_img = ImageTk.PhotoImage(image=pil_img)

        label_widget.config(image=tk_img)
        label_widget.image = tk_img


if __name__ == "__main__":
    try:
        root = tk.Tk()
        style = ttk.Style()
        style.theme_use('clam')

        app = App(root)
        root.mainloop()
    except Exception as e:
        print(f"Ocurrió un error fatal: {e}")