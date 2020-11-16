# Importando os pacotes necessários
from argparse import ArgumentParser
import cv2
import dlib
import imutils
import numpy as np
import time
from imutils.video import VideoStream
from imutils.video import FPS
from scipy.spatial import distance as dist
from collections import OrderedDict


class TrackableObject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]
        self.counted = False


class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.maxDisappeared = maxDisappeared

        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()

            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects


class CrowdControl:
    def __init__(self):
        # Analise dos argumentos da linha de comando
        self.args = ArgumentParser()
        self.args.add_argument("-i", "--input", type=str,
                        help="caminho para o vídeo de entrada")
        self.args.add_argument("-o", "--output", type=str,
                        help="caminho para o vídeo de saída")
        self.args = vars(self.args.parse_args())

        # Inicializa a lista de classes que o modelo MobileNet SSD foi treinado para detectar
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]

        # carrega os modelos serializados do disco
        print("carregando modelo...")
        net = cv2.dnn.readNetFromCaffe(
            "mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

        # Se o caminho do vídeo não for especificado, muda o stream de vídeo para webcam
        if not self.args.get("input", False):
            print("inicializando video stream...")
            vs = VideoStream(src=0).start()
            time.sleep(2.0)

        # caso contrário abra o vídeo especificado
        else:
            print("abrindo arquivo de vídeo...")
            vs = cv2.VideoCapture(self.args["input"])

        # Inicializa o ponteiro de escrita de vídeo
        writer = None

        # Inicializa as dimensões do frame
        W = None
        H = None

        # Instancia o rastreador do centroid, então inicialisa uma lista para armazenar
        # cada uma dos rastreadores correlacionados da dlib, seguindo de um
        # dicionário mapeando cada um dos objetos únicos para um objeto do tipo TrackableObject
        ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        trackers = []
        trackableObjects = {}

        # Inicializa o número total de frames processados até então, além
        # do número total de objetos além dos que se moveram para cima ou para baixo
        totalFrames = 0
        totalDown = 0
        totalUp = 0

        # Inicializa a estimativa de frames por segundo
        fps = FPS().start()

        # Loop principal, que passa pelos frames do stream de vídeo
        while True:
            # Captura o próximo frame e toma decisões diferentes se a fonte dos frames
            # for VideoCapture ou VideoStream

            frame = vs.read()
            frame = frame[1] if self.args.get("input", False) else frame

            # Se o vídeo esta sendo reproduzido e não foi possível pegar o próximo frame
            # então o vídeo chegou ao final

            if self.args["input"] is not None and frame is None:
                break

            # redimensiona o frame para o máximo de 500 pixels ( quanto menos dados para
            # analisar, mais rápida será a analise ), então converte o frame do formato
            # BGR para RGB para futura analise com dlib

            frame = imutils.resize(frame, width=500)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Caso as dimensões do frame forem vazias, ajuste
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # Se o vídeo será gravado em disco, inialize o ponteiro de escrita
            if self.args["output"] is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(self.args["output"], fourcc, 30,
                                         (W, H), True)

            # Inicializa o estatus corrente do processamento
            status = "Aguardando"
            rects = []

            # Verifica se o algorítimo é adequado para para o rastreamento dos objetos
            # detectados
            if totalFrames % 30 == 0:
                # Configura o status e inicializa uma lista de rastreadores
                status = "Detectando"
                trackers = []

                # Converte o frame para um blob e passe esse blob para a rede neural
                # e obtem a direção do movimento
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
                net.setInput(blob)
                detections = net.forward()

                # Roda o loop para as detecções
                for i in np.arange(0, detections.shape[2]):
                    # extrai o nível de confiança associada a previsão
                    confidence = detections[0, 0, i, 2]

                    # filtra detecções fracas exigindo um nível mínimo de confiança
                    if confidence > 0.4:
                        # extrai o índice das classes da lista de detecção
                        idx = int(detections[0, 0, i, 1])

                        # Se a classe não for uma pessoa ignore.
                        if CLASSES[idx] != "person":
                            continue

                        # calcula as coordenadas (x, y) da caixa de colisão para cada
                        # objeto

                        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = box.astype("int")

                        # constroi a abstração geométrica da correlação do dlib
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(rgb, rect)

                        # adiciona o rastreador na lista de rastreadores
                        trackers.append(tracker)

            else:
                # loop para os rastreadores
                for tracker in trackers:
                    status = "Rastreando"

                    # atualiza o rastreador e armazena posição
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    # armazena posição do objeto
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())

                    # adiciona posição do objeto para o array de colisões
                    rects.append((startX, startY, endX, endY))

            # desenha uma linha horizontal no centro do frame, uma vez que um objeto cruza
            # a linha ele é determinado se ele esta entrando ou saindo
            cv2.line(frame, (0, H // 2),
                     (W, H // 2), (0, 255, 255), 2)

            # Usar o rastreador do rastreador do centroid para associar o centroid antigo
            # com os centroids novos recém computados
            objects = ct.update(rects)

            # loop pelos objetos rastreados
            for (objectID, centroid) in objects.items():
                # verifica que o objeto rastreado existe para o objeto atual
                to = trackableObjects.get(objectID, None)

                # se não existe o objeto rastreado, crie um
                if to is None:
                    to = TrackableObject(objectID, centroid)

                # caso contrário, existe um objeto rastreado que podemos usar para determinar a direção
                else:
                    # a diferença entre a coordernada y do centroid atual e do centroid
                    # anterior nos ajuda a determinar se o objeto esta se movendo para
                    # dentro do espaço público ou para fora

                    x = [c[1] for c in to.centroids]
                    #import pdb; pdb.set_trace()
                    direction = centroid[1] - np.mean(x)

                    to.centroids.append(centroid)

                    # verifica se o objeto já foi contado ou não
                    if not to.counted:
                        # Se a direção é negativa (indica que o objeto se move para cima)
                        # e o centroid está acima da linha central, conte o objeto

                        if direction < 0 and centroid[1] < H // 2:
                            totalUp += 1
                            to.counted = True

                        # Se a direção é positiva (indica que o objeto se move para baixo)
                        # is moving down) e o centroid está abaixo da linha central,
                        # conte o objeto
                        elif direction > 0 and centroid[1] > H // 2:
                            totalDown += 1
                            to.counted = True

                # registre o objeto rastreado no dicionário
                trackableObjects[objectID] = to

                # desenher tanto o ID do objeto e o centroid no frame de saída
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(
                    frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # constroi uma tupla com as informações mostradas no frame
            info = [
                ("Entrando", totalUp),
                ("Saindo", totalDown),
                ("Lotacao", totalUp - totalDown),
            ]

            # itera sobre o conteúdo da tupla e desenha no frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # verifique se o frame deve ser gravado em disco
            if writer is not None:
                writer.write(frame)

            # mostra o frame de saída
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # se a tecla `q` for pressionado quebre o loop
            if key == ord("q"):
                break

            # incrementa o número total de frames processados e atualiza o contador de FPS
            totalFrames += 1
            fps.update()

        # pare o timer e mostre o FPS
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # verifique se é necessário liberar o ponteiro de escrita
        if writer is not None:
            writer.release()

        # se um arquivo de vídeo não esta sendo utilizado pare o stream de vídeo
        if not self.args.get("input", False):
            vs.stop()

        # caso contrário, libere o ponteiro de escrita do vídeo
        else:
            vs.release()

        # feche todas as janelas
        cv2.destroyAllWindows()

CrowdControl()