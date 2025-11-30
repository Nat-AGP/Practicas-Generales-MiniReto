# Procesamiento Digital de Imágenes (PDI)

Este repositorio contiene tres prácticas completas de Procesamiento Digital de Imágenes que implementan diferentes técnicas y algoritmos fundamentales en el área.

## 📋 Prácticas Implementadas

### 1. **Práctica 1: Extracción de Componentes Conexas**
**Archivo:** `Practica 1 Minireto - Extracción de Componentes Conexas.py`

#### Funcionalidades:
- **Operaciones aritméticas**: Suma, resta y multiplicación de imágenes
- **Operaciones lógicas**: AND, OR, XOR, NOT con máscaras personalizadas
- **Umbralización avanzada**: Múltiples métodos (BINARIO, BINARIO_INV, TRUNC, TOZERO, etc.)
- **Componentes conexas**: Etiquetado y análisis con `connectedComponentsWithStats`
- **Análisis de histograma**: Ecualización y CLAHE
- **Visualización completa**: Generación de reportes gráficos integrados

#### Características:
- Interfaz por consola interactiva
- Carga automática de imágenes
- Guardado automático de resultados
- Visualización con matplotlib

---

### 2. **Práctica 2: Mejoramiento de Imagen**
**Archivo:** `Práctica 2 Minireto - Mejoramiento de una imagen.py`

#### Funcionalidades:

##### **Modo Restauración:**
- **Generación de ruido**: Sal y pimienta, Gaussiano
- **Filtros de restauración**: 
  - Mediana (para ruido sal y pimienta)
  - Gaussiano (suavizado)
  - Promedio (blur)

##### **Modo Filtros Pasaaltas:**
- **Detección de bordes**:
  - Roberts
  - Sobel
  - Prewitt
  - Canny
  - Kirsch
  - Laplaciano

#### Características:
- Interfaz gráfica moderna con Tkinter
- Visualización en tiempo real con 3 paneles
- Procesamiento en color y escala de grises
- Controles interactivos deslizantes

---

### 3. **Práctica 3: Segmentación de Regiones**
**Archivo:** `Práctica 3 Minireto - Segmentación de regiones.py`

#### Funcionalidades:

##### **Ajuste de Brillo:**
- Desplazamiento (+/- brillo)
- Expansión y contracción de histograma
- Corrección gamma
- **Ecualizaciones avanzadas**:
  - Uniforme
  - Exponencial
  - Rayleigh
  - Hipercúbica
  - Logarítmica hiperbólica

##### **Segmentación:**
- **Métodos de umbralizado**:
  - Otsu
  - Entropía de Kapur
  - Mínimo del histograma
  - Media
  - Umbral en banda (80-150)

#### Características:
- Integración de matplotlib en Tkinter
- Visualización de histogramas con umbrales
- Procesamiento en tiempo real
- Interfaz profesional con temas modernos

## 🛠️ Tecnologías Utilizadas

- **Python 3.x**
- **OpenCV** (cv2) - Procesamiento de imágenes
- **NumPy** - Cálculos numéricos
- **Matplotlib** - Visualización y gráficos
- **Tkinter** - Interfaces gráficas
- **SciPy** - Algoritmos avanzados
- **PIL (Pillow)** - Manipulación de imágenes

## 📅 Planeer 
📌[Planner Fase 1]()
📌[Planner Fase 2]()
📌[Planner Fase 3]()

## 🚀 Cómo Ejecutar

1. **Instalar dependencias:**
```bash
pip install opencv-python numpy matplotlib pillow scipy
