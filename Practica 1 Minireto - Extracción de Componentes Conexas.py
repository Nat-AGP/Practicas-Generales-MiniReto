import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Variables globales para almacenar la imagen y resultados
img2 = None
gray_img2 = None
threshold_img2 = None
resultados_guardados = {}

# =============================================================================
# CARGA AUTOMÁTICA DE IMAGEN
# =============================================================================

def cargar_imagen_automatico():
    """Cargar automáticamente la imagen 2 al iniciar el programa"""
    global img2, gray_img2, threshold_img2
    
    # Definir la ruta de la imagen 2
    ruta_imagen2 = r"C:\Users\natal\OneDrive\Documentos\Visual Estudio ESCOM\PDI\imagen2.jpg.jpg"

    print("="*50)
    print("CARGANDO IMAGEN 2 AUTOMÁTICAMENTE")
    print("="*50)
    print(f"Ruta: {ruta_imagen2}")

    # Cargar imagen desde la ruta especificada
    img2 = cv2.imread(ruta_imagen2)

    # Verificar que la imagen se cargó correctamente
    if img2 is None:
        print("❌ ERROR: No se pudo cargar la imagen 2. Creando imagen de ejemplo...")
        # Crear una imagen de ejemplo si no existe
        img2 = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(img2, (100, 100), (300, 300), (255, 255, 255), -1)
        cv2.circle(img2, (200, 200), 80, (200, 200, 200), -1)
        cv2.line(img2, (50, 200), (350, 200), (150, 150, 150), 10)
        cv2.putText(img2, 'EJEMPLO', (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        print("✅ Imagen 2 cargada correctamente")

    # Procesamiento inicial
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, threshold_img2 = cv2.threshold(gray_img2, 127, 255, cv2.THRESH_BINARY)
    
    print(f"📐 Dimensiones: {img2.shape}")
    print("✅ Procesamiento inicial completado (escala de grises + umbralización)")
    
    return True

# =============================================================================
# FUNCIONES DE OPERACIONES CON UNA SOLA IMAGEN
# =============================================================================

def operaciones_aritmeticas():
    """Realizar operaciones aritméticas con la imagen 2"""
    if img2 is None:
        print("❌ Error: La imagen no está cargada")
        return
    
    print("\n" + "="*50)
    print("OPERACIONES ARITMÉTICAS")
    print("="*50)
    
    # Operaciones con escalares
    suma = cv2.add(img2, 50)
    resta = cv2.subtract(img2, 50)
    multiplicacion = cv2.multiply(img2, 1.5)
    
    # Visualización
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Imagen Original')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(suma, cv2.COLOR_BGR2RGB))
    plt.title('Suma +50')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(resta, cv2.COLOR_BGR2RGB))
    plt.title('Resta -50')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(multiplicacion, cv2.COLOR_BGR2RGB))
    plt.title('Multiplicación ×1.5')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Guardar resultados
    resultados_guardados['suma'] = suma
    resultados_guardados['resta'] = resta
    resultados_guardados['multiplicacion'] = multiplicacion
    
    print("✅ Operaciones aritméticas completadas")
    
    input("\nPresiona Enter para continuar...")

def operaciones_logicas():
    """Realizar operaciones lógicas con la imagen 2"""
    if img2 is None:
        print("❌ Error: La imagen no está cargada")
        return
    
    print("\n" + "="*50)
    print("OPERACIONES LÓGICAS")
    print("="*50)
    
    # Crear diferentes máscaras para operaciones lógicas
    altura, ancho = gray_img2.shape
    
    # Máscara 1: Circular en el centro
    mascara_circular = np.zeros((altura, ancho), dtype=np.uint8)
    centro = (ancho // 2, altura // 2)
    radio = min(ancho, altura) // 4
    cv2.circle(mascara_circular, centro, radio, 255, -1)
    
    # Máscara 2: Rectangular
    mascara_rectangular = np.zeros((altura, ancho), dtype=np.uint8)
    x1, y1 = ancho // 4, altura // 4
    x2, y2 = 3 * ancho // 4, 3 * altura // 4
    cv2.rectangle(mascara_rectangular, (x1, y1), (x2, y2), 255, -1)
    
    # Operaciones lógicas con máscaras
    and_img = cv2.bitwise_and(gray_img2, gray_img2, mask=mascara_circular)
    or_img = cv2.bitwise_or(gray_img2, gray_img2, mask=mascara_rectangular)
    xor_img = cv2.bitwise_xor(gray_img2, 128, mask=mascara_circular)
    
    # Operación NOT
    not_img = cv2.bitwise_not(gray_img2)
    
    # Visualización
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 4, 1)
    plt.imshow(gray_img2, cmap='gray')
    plt.title('Imagen Original Gris')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(mascara_circular, cmap='gray')
    plt.title('Máscara Circular')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(mascara_rectangular, cmap='gray')
    plt.title('Máscara Rectangular')
    plt.axis('off')
    
    plt.subplot(2, 4, 4)
    plt.imshow(not_img, cmap='gray')
    plt.title('NOT (Inversión)')
    plt.axis('off')
    
    plt.subplot(2, 4, 5)
    plt.imshow(and_img, cmap='gray')
    plt.title('AND con Máscara Circular')
    plt.axis('off')
    
    plt.subplot(2, 4, 6)
    plt.imshow(or_img, cmap='gray')
    plt.title('OR con Máscara Rectangular')
    plt.axis('off')
    
    plt.subplot(2, 4, 7)
    plt.imshow(xor_img, cmap='gray')
    plt.title('XOR con 128')
    plt.axis('off')
    
    plt.subplot(2, 4, 8)
    plt.imshow(threshold_img2, cmap='gray')
    plt.title('Umbralizada Original')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Guardar resultados
    resultados_guardados['and_img'] = and_img
    resultados_guardados['or_img'] = or_img
    resultados_guardados['not_img'] = not_img
    
    print("✅ Operaciones lógicas completadas")
    
    input("\nPresiona Enter para continuar...")

def umbralizacion_avanzada():
    """Aplicar diferentes técnicas de umbralización"""
    if img2 is None:
        print("❌ Error: La imagen no está cargada")
        return
    
    print("\n" + "="*50)
    print("UMBRALIZACIÓN AVANZADA")
    print("="*50)
    
    # Probar diferentes valores de umbralización
    umbrales = [100, 127, 150, 200]
    
    # Diferentes tipos de umbralización
    _, binaria = cv2.threshold(gray_img2, 127, 255, cv2.THRESH_BINARY)
    _, bin_inv = cv2.threshold(gray_img2, 127, 255, cv2.THRESH_BINARY_INV)
    _, trunc = cv2.threshold(gray_img2, 127, 255, cv2.THRESH_TRUNC)
    _, tozero = cv2.threshold(gray_img2, 127, 255, cv2.THRESH_TOZERO)
    _, tozero_inv = cv2.threshold(gray_img2, 127, 255, cv2.THRESH_TOZERO_INV)
    
    # Umbralización adaptativa
    adaptativa_gauss = cv2.adaptiveThreshold(gray_img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
    
    # Crear imágenes con diferentes umbrales
    imagenes_umbrales = []
    for umbral in umbrales:
        _, img_umbral = cv2.threshold(gray_img2, umbral, 255, cv2.THRESH_BINARY)
        imagenes_umbrales.append(img_umbral)
    
    # Visualización 
    plt.figure(figsize=(15, 12))
    
    # Fila 1
    plt.subplot(4, 3, 1)
    plt.imshow(gray_img2, cmap='gray')
    plt.title('Original Grises')
    plt.axis('off')
    
    plt.subplot(4, 3, 2)
    plt.imshow(binaria, cmap='gray')
    plt.title('BINARIO')
    plt.axis('off')
    
    plt.subplot(4, 3, 3)
    plt.imshow(bin_inv, cmap='gray')
    plt.title('BINARIO_INV')
    plt.axis('off')
    
    # Fila 2
    plt.subplot(4, 3, 4)
    plt.imshow(trunc, cmap='gray')
    plt.title('TRUNC')
    plt.axis('off')
    
    plt.subplot(4, 3, 5)
    plt.imshow(tozero, cmap='gray')
    plt.title('TOZERO')
    plt.axis('off')
    
    plt.subplot(4, 3, 6)
    plt.imshow(tozero_inv, cmap='gray')
    plt.title('TOZERO_INV')
    plt.axis('off')
    
    # Fila 3
    plt.subplot(4, 3, 7)
    plt.imshow(adaptativa_gauss, cmap='gray')
    plt.title('ADAPTATIVA GAUSSIANA')
    plt.axis('off')
    
    # Mostrar diferentes umbrales en las posiciones restantes
    for i, (umbral, img_umbral) in enumerate(zip(umbrales, imagenes_umbrales)):
        plt.subplot(4, 3, 8 + i)
        plt.imshow(img_umbral, cmap='gray')
        plt.title(f'Umbral {umbral}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Guardar resultados
    resultados_guardados['umbral_adaptativo'] = adaptativa_gauss
    
    print("✅ Umbralización avanzada completada")
    
    input("\nPresiona Enter para continuar...")

def componentes_conexas():
    """Aplicar etiquetado de componentes conexas"""
    if img2 is None:
        print("❌ Error: La imagen no está cargada")
        return
    
    print("\n" + "="*50)
    print("COMPONENTES CONEXAS")
    print("="*50)
    
    # Aplicar etiquetado de componentes conexas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold_img2, connectivity=8)
    
    # Crear una imagen de componentes conexas coloreada
    labels_colored = np.uint8(255 * labels / np.max(labels))
    labels_colored = cv2.applyColorMap(labels_colored, cv2.COLORMAP_JET)
    
    # Crear imagen con bounding boxes
    img_with_boxes = img2.copy()
    
    print(f"📊 Se encontraron {num_labels - 1} componentes conexas (excluyendo fondo)")
    
    # Dibujar bounding boxes y mostrar información
    for i in range(1, num_labels):  # Empezar desde 1 para excluir el fondo
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Dibujar rectángulo
        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Mostrar información del componente
        print(f"  Componente {i}: Posición ({x}, {y}), Tamaño {w}x{h}, Área: {area} píxeles")
    
    # Visualización
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(threshold_img2, cmap='gray')
    plt.title('Imagen Umbralizada')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(labels_colored, cv2.COLOR_BGR2RGB))
    plt.title(f'Componentes Conexas ({num_labels-1} componentes)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title('Bounding Boxes')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    resultados_guardados['componentes_conexas'] = labels_colored
    
    print("✅ Análisis de componentes conexas completado")
    
    input("\nPresiona Enter para continuar...")

def analisis_histograma():
    """Realizar análisis de histograma y ajuste de brillo"""
    if img2 is None:
        print("❌ Error: La imagen no está cargada")
        return
    
    print("\n" + "="*50)
    print("ANÁLISIS DE HISTOGRAMA")
    print("="*50)
    
    # Calcular histograma
    histograma = cv2.calcHist([gray_img2], [0], None, [256], [0, 256])
    
    # Aplicar diferentes técnicas de mejora
    img_ecualizada = cv2.equalizeHist(gray_img2)
    histograma_ecualizado = cv2.calcHist([img_ecualizada], [0], None, [256], [0, 256])
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(gray_img2)
    histograma_clahe = cv2.calcHist([img_clahe], [0], None, [256], [0, 256])
    
    # Umbralizar después de las mejoras
    _, threshold_ecualizada = cv2.threshold(img_ecualizada, 127, 255, cv2.THRESH_BINARY)
    _, threshold_clahe = cv2.threshold(img_clahe, 127, 255, cv2.THRESH_BINARY)
    
    # Visualización completa
    plt.figure(figsize=(15, 10))
    
    # Fila 1: Imágenes
    plt.subplot(2, 4, 1)
    plt.imshow(gray_img2, cmap='gray')
    plt.title('Original Gris')
    plt.axis('off')
    
    plt.subplot(2, 4, 2)
    plt.imshow(img_ecualizada, cmap='gray')
    plt.title('Ecualizada')
    plt.axis('off')
    
    plt.subplot(2, 4, 3)
    plt.imshow(img_clahe, cmap='gray')
    plt.title('CLAHE')
    plt.axis('off')
    
    plt.subplot(2, 4, 4)
    plt.imshow(threshold_img2, cmap='gray')
    plt.title('Umbral Original')
    plt.axis('off')
    
    # Fila 2: Histogramas y resultados
    plt.subplot(2, 4, 5)
    plt.plot(histograma)
    plt.title('Histograma Original')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')
    
    plt.subplot(2, 4, 6)
    plt.plot(histograma_ecualizado)
    plt.title('Histograma Ecualizado')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')
    
    plt.subplot(2, 4, 7)
    plt.plot(histograma_clahe)
    plt.title('Histograma CLAHE')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')
    
    plt.subplot(2, 4, 8)
    plt.imshow(threshold_ecualizada, cmap='gray')
    plt.title('Umbral Ecualizada')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Guardar resultados
    resultados_guardados['img_ecualizada'] = img_ecualizada
    resultados_guardados['img_clahe'] = img_clahe
    
    print("✅ Análisis de histograma completado")
    
    input("\nPresiona Enter para continuar...")

def mostrar_todos_resultados():
    """Mostrar todos los resultados en una visualización completa"""
    if img2 is None:
        print("❌ Error: La imagen no está cargada")
        return
    
    print("\n" + "="*50)
    print("MOSTRAR TODOS LOS RESULTADOS")
    print("="*50)
    
    # Realizar operaciones para la visualización completa
    altura, ancho = gray_img2.shape
    
    # Crear máscara circular
    mascara_circular = np.zeros((altura, ancho), dtype=np.uint8)
    cv2.circle(mascara_circular, (ancho//2, altura//2), min(ancho, altura)//4, 255, -1)
    
    # Operaciones aritméticas
    suma = cv2.add(img2, 50)
    multiplicacion = cv2.multiply(img2, 1.5)
    
    # Operaciones lógicas
    and_img = cv2.bitwise_and(gray_img2, gray_img2, mask=mascara_circular)
    not_img = cv2.bitwise_not(gray_img2)
    
    # Umbral adaptativo
    umbral_adaptativo = cv2.adaptiveThreshold(gray_img2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
    
    # Componentes conexas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(threshold_img2, connectivity=8)
    labels_colored = np.uint8(255 * labels / np.max(labels))
    labels_colored = cv2.applyColorMap(labels_colored, cv2.COLORMAP_JET)
    
    # Visualización completa
    plt.figure(figsize=(20, 12))
    
    # Fila 1: Imágenes base y operaciones aritméticas
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('Original Color')
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(gray_img2, cmap='gray')
    plt.title('Escala Grises')
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(cv2.cvtColor(suma, cv2.COLOR_BGR2RGB))
    plt.title('Suma +50')
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(cv2.cvtColor(multiplicacion, cv2.COLOR_BGR2RGB))
    plt.title('Multiplicación ×1.5')
    plt.axis('off')
    
    # Fila 2: Operaciones lógicas y umbralización
    plt.subplot(3, 4, 5)
    plt.imshow(and_img, cmap='gray')
    plt.title('AND con Máscara')
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(not_img, cmap='gray')
    plt.title('NOT (Inversión)')
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(threshold_img2, cmap='gray')
    plt.title('Umbral Simple')
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(umbral_adaptativo, cmap='gray')
    plt.title('Umbral Adaptativo')
    plt.axis('off')
    
    # Fila 3: Componentes conexas y análisis avanzado
    plt.subplot(3, 4, 9)
    plt.imshow(cv2.cvtColor(labels_colored, cv2.COLOR_BGR2RGB))
    plt.title(f'Componentes Conexas\n({num_labels-1} objetos)')
    plt.axis('off')
    
    # Histograma
    histograma = cv2.calcHist([gray_img2], [0], None, [256], [0, 256])
    plt.subplot(3, 4, 10)
    plt.plot(histograma)
    plt.title('Histograma Original')
    plt.xlabel('Intensidad')
    plt.ylabel('Frecuencia')
    
    # Bordes
    bordes = cv2.Canny(gray_img2, 100, 200)
    plt.subplot(3, 4, 11)
    plt.imshow(bordes, cmap='gray')
    plt.title('Detección de Bordes')
    plt.axis('off')
    
    # Imagen con diferente umbral
    _, threshold_alto = cv2.threshold(gray_img2, 200, 255, cv2.THRESH_BINARY)
    plt.subplot(3, 4, 12)
    plt.imshow(threshold_alto, cmap='gray')
    plt.title('Umbral Alto (200)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('resultados_completos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualización completa generada y guardada como 'resultados_completos.png'")
    input("\nPresiona Enter para continuar...")

def guardar_resultados():
    """Guardar todos los resultados en archivos"""
    if not resultados_guardados:
        print("❌ No hay resultados para guardar. Ejecuta algunas operaciones primero.")
        return
    
    print("\n" + "="*50)
    print("GUARDAR RESULTADOS")
    print("="*50)
    
    # Crear directorio para resultados si no existe
    if not os.path.exists('resultados_imagen2'):
        os.makedirs('resultados_imagen2')
    
    # Guardar imagen original procesada
    cv2.imwrite('resultados_imagen2/imagen2_original.jpg', img2)
    cv2.imwrite('resultados_imagen2/imagen2_gris.jpg', gray_img2)
    cv2.imwrite('resultados_imagen2/imagen2_umbral.jpg', threshold_img2)
    
    # Guardar resultados individuales
    for nombre, imagen in resultados_guardados.items():
        ruta = f'resultados_imagen2/{nombre}.jpg'
        cv2.imwrite(ruta, imagen)
        print(f"✅ Guardado: {ruta}")
    
    print(f"\n📁 Se guardaron {len(resultados_guardados) + 3} archivos en 'resultados_imagen2/'")
    input("\nPresiona Enter para continuar...")

# =============================================================================
# MENÚ PRINCIPAL (SIN REFLEXIONES)
# =============================================================================

def mostrar_menu():
    """Mostrar el menú principal"""
    print("\n" + "="*60)
    print("PRÁCTICA 1: TRANSFORMACIONES LÓGICAS EN IMÁGENES DIGITALES")
    print("="*60)
    print("1. Operaciones aritméticas en imágenes")
    print("2. Operaciones lógicas en imágenes") 
    print("3. Umbralización avanzada")
    print("4. Componentes conexas")
    print("5. Análisis de histograma")
    print("6. Mostrar todos los resultados")
    print("7. Guardar resultados")
    print("0. Salir")
    print("-"*60)

def main():
    """Función principal del programa"""
    print("Bienvenido al sistema de procesamiento de imágenes")
    print("Práctica 1: Transformaciones Lógicas en Imágenes Digitales")
    
    # Cargar imagen automáticamente al iniciar
    if cargar_imagen_automatico():
        print("\n✅ Imagen 2 cargada y preprocesada automáticamente")
        print("📊 Listo para realizar operaciones...")
    else:
        print("❌ Error al cargar la imagen")
        return
    
    input("\nPresiona Enter para continuar al menú principal...")
    
    while True:
        mostrar_menu()
        opcion = input("Selecciona una opción (0-7): ")
        
        if opcion == '1':
            operaciones_aritmeticas()
        elif opcion == '2':
            operaciones_logicas()
        elif opcion == '3':
            umbralizacion_avanzada()
        elif opcion == '4':
            componentes_conexas()
        elif opcion == '5':
            analisis_histograma()
        elif opcion == '6':
            mostrar_todos_resultados()
        elif opcion == '7':
            guardar_resultados()
        elif opcion == '0':
            print("\n¡Gracias por usar el sistema de procesamiento de imágenes!")
            print("Hasta pronto! 👋")
            break
        else:
            print("❌ Opción no válida. Por favor, selecciona una opción del 0 al 7.")
            input("Presiona Enter para continuar...")

# =============================================================================
# EJECUCIÓN DEL PROGRAMA
# =============================================================================

if __name__ == "__main__":
    main()