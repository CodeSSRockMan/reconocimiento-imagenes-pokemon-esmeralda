import cv2
import os
import json
import sys
import uuid
from datetime import datetime

# Función para verificar si un archivo existe
def check_file_path(file_path):
    return os.path.exists(file_path)

def load_images(base_dir, capture_data, debug_dir="debug"):
    """
    Carga las imágenes desde un archivo JSON y las procesa.
    """
    images = {}

    for capture in capture_data:
        # Construir la ruta completa de la imagen
        image_path = os.path.join(base_dir, capture["ruta"])
        print(f"[DEBUG] Procesando captura: {capture['id']} - Ruta completa: {image_path}")  # Depuración

        # Verifica si la ruta existe
        if check_file_path(image_path):
            print(f"[DEBUG] El archivo existe: {image_path}")  # Depuración
            image = cv2.imread(image_path)

            if image is None:
                print(f"[ERROR] No se pudo cargar la imagen: {image_path}")  # Depuración
                continue

            # Convertir a escala de grises
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Guardar en el directorio de depuración
            ref_debug_path = os.path.join(debug_dir, f"{capture['id']}_{uuid.uuid4().hex[:8]}.png")
            cv2.imwrite(ref_debug_path, gray_image)
            print(f"[DEBUG] Imagen guardada en escala de grises en: {ref_debug_path}")

            images[capture["id"]] = gray_image
        else:
            print(f"[DEBUG] Error: El archivo {image_path} no existe.")  # Depuración
            continue  # Continúa procesando las demás imágenes

    if not images:
        print("[WARNING] No se cargaron imágenes.")  # Depuración
    return images

def load_reference_images_from_json(json_data, base_dir, debug_dir="debug"):
    """
    Carga las imágenes de referencia desde el archivo JSON.
    """
    reference_images = {}

    for state, state_data in json_data.items():
        print(f"[DEBUG] Procesando estado: {state}")  # Depuración
        images = load_images(base_dir, state_data["capturas"], debug_dir)
        if images:
            reference_images[state] = images
            print(f"[DEBUG] Imágenes de referencia para estado {state} cargadas.")  # Depuración
        else:
            print(f"[WARNING] No se cargaron imágenes para el estado: {state}")  # Depuración

        # Procesar subestados si existen
        if "subestados" in state_data:
            for substate, substate_data in state_data["subestados"].items():
                substate_key = f"{state}_{substate}"
                print(f"[DEBUG] Procesando subestado recursivo: {substate_key}")  # Depuración
                sub_images = load_images(base_dir, substate_data["capturas"], debug_dir)
                if sub_images:
                    reference_images[substate_key] = sub_images
                    print(f"[DEBUG] Imágenes de referencia para subestado {substate_key} cargadas.")  # Depuración
                else:
                    print(f"[WARNING] No se cargaron imágenes para el subestado: {substate_key}")  # Depuración

                # Procesar sub-subestados si existen
                if "subestados" in substate_data:
                    for sub_substate, sub_substate_data in substate_data["subestados"].items():
                        sub_substate_key = f"{substate_key}_{sub_substate}"
                        print(f"[DEBUG] Procesando sub-subestado recursivo: {sub_substate_key}")  # Depuración
                        sub_sub_images = load_images(base_dir, sub_substate_data["capturas"], debug_dir)
                        if sub_sub_images:
                            reference_images[sub_substate_key] = sub_sub_images
                            print(f"[DEBUG] Imágenes de referencia para sub-subestado {sub_substate_key} cargadas.")  # Depuración
                        else:
                            print(f"[WARNING] No se cargaron imágenes para el sub-subestado: {sub_substate_key}")  # Depuración

    if not reference_images:
        print("[WARNING] No se cargaron imágenes de referencia.")  # Depuración
    else:
        print("[DEBUG] Carga de imágenes de referencia completada.")  # Depuración
    return reference_images

def detect_game_state(capture_image, reference_images, debug_dir="debug"):
    """
    Detecta el estado principal de una captura de pantalla utilizando Ratio Test y un umbral de distancia ajustado.
    """
    # Crear directorio de depuración si no existe
    os.makedirs(debug_dir, exist_ok=True)

    # Convertir la captura a escala de grises
    gray_capture = cv2.cvtColor(capture_image, cv2.COLOR_BGR2GRAY)

    # Guardar la captura en escala de grises con un nombre único
    capture_debug_path = os.path.join(debug_dir, f"capture_gray_{uuid.uuid4().hex[:8]}.png")
    cv2.imwrite(capture_debug_path, gray_capture)
    print(f"[DEBUG] Captura en escala de grises guardada en: {capture_debug_path}")

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray_capture, None)

    # Paso 1: Detectar el estado principal
    for state, state_info in reference_images.items():
        if isinstance(state_info, dict):  # Asegurarse de que estamos manejando la lista de imágenes de referencia
            for ref_id, ref_image in state_info.items():
                kp2, des2 = orb.detectAndCompute(ref_image, None)

                # Comparar la referencia con la captura
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)  # No usar crossCheck aquí
                matches = bf.knnMatch(des1, des2, k=2)  # Buscar las 2 mejores coincidencias por punto

                # Aplicar Ratio Test para filtrar malas coincidencias
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:  # Usar el Ratio Test
                        good_matches.append(m)

                # Verificar si hay suficientes coincidencias
                if len(good_matches) > 15:  # Ajustar según sea necesario
                    # Guardar una visualización de las coincidencias con un nombre único
                    debug_match_path = os.path.join(debug_dir, f"{state}_match_{uuid.uuid4().hex[:8]}.png")
                    match_img = cv2.drawMatches(gray_capture, kp1, ref_image, kp2, good_matches, None, flags=2)
                    cv2.imwrite(debug_match_path, match_img)
                    print(f"[DEBUG] Coincidencias guardadas en: {debug_match_path}")

                    # Mostrar información de coincidencias
                    print(f"[DEBUG] Comparando captura con {ref_image}")
                    print(f"[DEBUG] Coincidencias encontradas: {len(good_matches)}")

                    if len(good_matches) > 20:  # Ajustar el umbral según sea necesario
                        print(f"[INFO] Estado detectado: {state}")
                        return state

    print("[INFO] Estado no reconocido, asignando a 'default'")
    return "default"

def save_categorized_image(capture_image, state, output_dir):
    """
    Guarda la imagen categorizada dentro de una subcarpeta del estado correspondiente.
    """
    state_parts = state.split("_")  # Suponiendo que los estados se nombran como "estado_subestado"
    
    # Crear la ruta de las subcarpetas
    state_dir = os.path.join(output_dir, *state_parts)
    os.makedirs(state_dir, exist_ok=True)
    
    # Guardar la imagen con un nombre único
    capture_name = f"{uuid.uuid4().hex[:8]}.png"
    capture_path = os.path.join(state_dir, capture_name)
    cv2.imwrite(capture_path, capture_image)
    print(f"[INFO] Imagen categorizada guardada en: {capture_path}")

def save_default_image(capture_image, output_dir):
    """
    Guarda la imagen que no se pudo categorizar en una carpeta por defecto.
    """
    default_dir = os.path.join(output_dir, "default")
    os.makedirs(default_dir, exist_ok=True)
    capture_name = f"{uuid.uuid4().hex[:8]}.png"
    capture_path = os.path.join(default_dir, capture_name)
    cv2.imwrite(capture_path, capture_image)
    print(f"[INFO] Imagen no categorizada guardada en: {capture_path}")

def load_json_data(file_path):
    return json.load(open(file_path, 'r')) if os.path.exists(file_path) else {}

def save_json_data(file_path, data):
    json.dump(data, open(file_path, 'w'), indent=4)

def process_images(image_files, reference_images, output_dir, json_file, log_file):
    json_data = load_json_data(json_file)
    if reference_images is None:
        print("Error al cargar imágenes de referencia. Finalizando el programa.")
        return
    total_images, successful_images, error_images = 0, 0, 0
    with open(log_file, 'a') as log:
        log.write(f"Iniciando procesamiento: {datetime.now()}\n")
        for image_file in image_files:
            total_images += 1
            capture_image = cv2.imread(image_file)
            if capture_image is None:
                error_images += 1
                continue
            state = detect_game_state(capture_image, reference_images)
            if state:
                save_categorized_image(capture_image, state, output_dir)
                successful_images += 1
            else:
                save_default_image(capture_image, output_dir)
                error_images += 1

            # Eliminar el archivo original después de procesarlo
            try:
                os.remove(image_file)
                print(f"[INFO] Archivo original eliminado: {image_file}")
            except Exception as e:
                print(f"[ERROR] No se pudo eliminar el archivo {image_file}: {e}")

            save_json_data(json_file, json_data)

        log.write(f"Total procesadas: {total_images}, Éxito: {successful_images}, Errores: {error_images}\n")

# Punto de entrada
if __name__ == "__main__":
    if len(sys.argv) == 1:  # No hay argumentos proporcionados
        print("Error: Debes proporcionar el nombre de la carpeta como argumento.")
        sys.exit(1)

    folder_dir = sys.argv[1]  # Primer argumento como nombre de la carpeta
    base_dir = os.path.abspath(folder_dir)  # Ruta absoluta de la carpeta
    json_file = os.path.join(base_dir, "referencias.json")  # Archivo referencias.json dentro de la carpeta
    log_file = os.path.join(base_dir, "procesamiento_log.txt")  # Archivo log
    output_dir = os.path.join(base_dir, "capturas")  # Carpeta de capturas

    # Verificar que el archivo referencias.json exista antes de proceder
    if not check_file_path(json_file):
        print(f"Error: El archivo {json_file} no existe.")
        sys.exit(1)

    # Cargar imágenes de referencia desde el JSON
    reference_images = load_reference_images_from_json(load_json_data(json_file), base_dir)

    # Procesar las imágenes en la carpeta
    image_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(".png")]
    process_images(image_files, reference_images, output_dir, json_file, log_file)
