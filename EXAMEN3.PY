import cv2
import cvlib as cv
import os

def detectar_y_filtrar_objetos():
    # Acceder a la webcam
    # cv2.VideoCapture(0) inicializa la cámara por defecto.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la webcam. Asegúrate de que no esté en uso y los permisos sean correctos.")
        return

    # Definir los objetos de interés que queremos detectar y contar.
    objetos_de_interes = ['person', 'cell phone', 'mouse']
    
    # Bandera para controlar si ya se guardó la imagen para evitar guardados repetidos.
    imagen_guardada = False

    print("Iniciando detección. Presiona 'q' en la ventana de video para salir.")

    while True:
        # Leer un fotograma de la webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el fotograma. Finalizando...")
            break

        # Usar cvlib para detectar objetos comunes en el fotograma.
        # Esto devuelve las coordenadas de las cajas, las etiquetas y la confianza.
        boxes, labels, confidences = cv.detect_common_objects(frame)

        # Reiniciar contadores para cada nuevo fotograma.
        contador_personas = 0
        contador_telefonos = 0
        contador_ratones = 0

        # Iterar sobre las detecciones para filtrar y dibujar solo los objetos de interés.
        for box, label, conf in zip(boxes, labels, confidences):
            # Solo procesar si la etiqueta está en nuestra lista de interés.
            if label in objetos_de_interes:
                # Convertir las coordenadas a enteros.
                x, y, w, h = map(int, box)

                # Dibujar un rectángulo verde alrededor del objeto.
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
                # Poner la etiqueta y la confianza encima del rectángulo.
                cv2.putText(frame, f'{label}: {conf:.2f}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Actualizar los contadores.
                if label == 'person':
                    contador_personas += 1
                elif label == 'cell phone':
                    contador_telefonos += 1
                elif label == 'mouse':
                    contador_ratones += 1
                print(f"persona{contador_personas} Celular{contador_telefonos} mouse {contador_ratones} ")

        # Mostrar el conteo de los objetos en la esquina superior izquierda del fotograma.
        conteo_texto = f"Personas: {contador_personas} | Celulares: {contador_telefonos} | Ratones: {contador_ratones}"
        cv2.putText(frame, conteo_texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2) # Texto blanco

        # Guardar automáticamente una imagen si se detectan 2 o más personas.
        # La imagen solo se guardará una vez (controlado por 'imagen_guardada').
        if contador_personas >= 2 and not imagen_guardada:
            nombre_archivo = 'mobilenet_filtro_resultado.jpg'
            cv2.imwrite(nombre_archivo, frame)
            print(f"¡Atención! Se detectaron {contador_personas} personas. Imagen guardada como '{nombre_archivo}'")
            imagen_guardada = True # Cambiar la bandera para no guardar de nuevo.
        elif contador_personas < 2:
            # Si el número de personas cae por debajo de 2, reiniciamos la bandera
            # Esto permite que la imagen se guarde de nuevo si las condiciones se cumplen más tarde.
            imagen_guardada = False

        # Mostrar el fotograma procesado en una ventana.
        cv2.imshow('Deteccion de Objetos Filtrados', frame)

        # Esperar 1 milisegundo por una tecla. Si es 'q', salir del bucle.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar la webcam y cerrar todas las ventanas de OpenCV al finalizar.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar_y_filtrar_objetos()