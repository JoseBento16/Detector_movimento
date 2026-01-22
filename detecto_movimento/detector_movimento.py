import cv2

# Abrir webcam
camera = cv2.VideoCapture(0)

# Ler o primeiro frame como referência
ret, frame_anterior = camera.read()
frame_anterior = cv2.cvtColor(frame_anterior, cv2.COLOR_BGR2GRAY)
frame_anterior = cv2.GaussianBlur(frame_anterior, (21, 21), 0)

# Variáveis para manter o quadrado
ultimo_retangulo = None
contador_sem_movimento = 0
LIMITE_FRAMES = 10  # Quantos frames o quadrado permanece

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Converter para cinza
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_cinza = cv2.GaussianBlur(frame_cinza, (21, 21), 0)

    # Diferença entre frames
    diferenca = cv2.absdiff(frame_anterior, frame_cinza)
    _, thresh = cv2.threshold(diferenca, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Encontrar contornos
    contornos, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    maior_contorno = None
    maior_area = 0

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > maior_area and area > 3000:
            maior_area = area
            maior_contorno = contorno

    # Se detectar movimento
    if maior_contorno is not None:
        x, y, w, h = cv2.boundingRect(maior_contorno)
        ultimo_retangulo = (x, y, w, h)
        contador_sem_movimento = 0
    else:
        contador_sem_movimento += 1
        if contador_sem_movimento > LIMITE_FRAMES:
            ultimo_retangulo = None

    # Desenhar retângulo
    if ultimo_retangulo is not None:
        x, y, w, h = ultimo_retangulo
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            "Movimento detectado",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Mostrar tela
    cv2.imshow("Detector de Movimento", frame)

    # Atualizar frame anterior
    frame_anterior = frame_cinza

    # Pressione Q para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Finalizar
camera.release()
cv2.destroyAllWindows()
