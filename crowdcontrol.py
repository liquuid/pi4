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


class TrackableObj:
    def __init__(self, objectID, centroid):
        self.objID = objectID
        self.centroids = [centroid]
        self.counted = False


class CentroidTrk:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        self.nxtObjectID = 0
        self.objs = OrderedDict()
        self.desaparecidos = OrderedDict()

        self.mxDisappeared = maxDisappeared

        self.mxDistance = maxDistance

    def deregister(self, objectID):
        del self.objs[objectID]
        del self.desaparecidos[objectID]

    def register(self, centroid):
        self.objs[self.nxtObjectID] = centroid
        self.desaparecidos[self.nxtObjectID] = 0
        self.nxtObjectID += 1

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.desaparecidos.keys()):
                self.desaparecidos[objectID] += 1
                if self.desaparecidos[objectID] > self.mxDisappeared:
                    self.deregister(objectID)
            return self.objs

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objs) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        else:
            objectIDs = list(self.objs.keys())
            objectCentroids = list(self.objs.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedCols = set()
            usedRows = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.mxDistance:
                    continue

                objectID = objectIDs[row]
                self.objs[objectID] = inputCentroids[col]
                self.desaparecidos[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.desaparecidos[objectID] += 1

                    if self.desaparecidos[objectID] > self.mxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objs


class CrowdControl:
    def __init__(self):
        # Analise dos argumentos da linha de comando
        self.argumentos = ArgumentParser()
        self.argumentos.add_argument("-i", "--input", type=str,
                                     help="caminho para o vídeo de entrada")
        self.argumentos.add_argument("-o", "--output", type=str,
                                     help="caminho para o vídeo de saída")
        self.argumentos = vars(self.argumentos.parse_args())

        # Inicializa a lista de classes que o modelo MobileNet SSD foi treinado para detectar
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]

        # carrega os modelos serializados do disco
        net = cv2.dnn.readNetFromCaffe(
            "mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

        # Se o caminho do vídeo não for especificado, muda o stream de vídeo para webcam
        if not self.argumentos.get("input", False):
            vs = VideoStream(src=0).start()
            time.sleep(2.0)

        # caso contrário abra o vídeo especificado
        else:
            vs = cv2.VideoCapture(self.argumentos["input"])

        # Inicializa o ponteiro de escrita de vídeo
        writer = None

        # Inicializa as dimensões do quadro
        W = None
        H = None

        # Instancia o rastreador do centroid, então inicialisa uma lista para armazenar
        # cada uma dos rastreadores correlacionados da dlib, seguindo de um
        # dicionário mapeando cada um dos objetos únicos para um objeto do tipo TrackableObj
        ct = CentroidTrk(maxDisappeared=40, maxDistance=50)
        trackers = []
        trackableObjects = {}

        # Inicializa o número total de frames processados até então, além
        # do número total de objetos além dos que se moveram para cima ou para baixo
        totalFrames, totalDown, totalUp = (0, 0, 0)

        # Inicializa a estimativa de frames por segundo
        fps = FPS().start()

        # Loop principal, que passa pelos frames do stream de vídeo
        while True:
            # Captura o próximo quadro e toma decisões diferentes se a fonte dos frames
            # for VideoCapture ou VideoStream

            quadro = vs.read()
            quadro = quadro[1] if self.argumentos.get("input", False) else quadro

            # Se o vídeo esta sendo reproduzido e não foi possível pegar o próximo quadro
            # então o vídeo chegou ao final

            if self.argumentos["input"] is not None and quadro is None:
                break

            # redimensiona o quadro para o máximo de 500 pixels ( quanto menos dados para
            # analisar, mais rápida será a analise ), então converte o quadro do formato
            # BGR para RGB para futura analise com dlib

            quadro = imutils.resize(quadro, width=500)
            rgb = cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)

            # Caso as dimensões do quadro forem vazias, ajuste
            if W is None or H is None:
                (H, W) = quadro.shape[:2]

            # Se o vídeo será gravado em disco, inialize o ponteiro de escrita
            if self.argumentos["output"] is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(self.argumentos["output"], fourcc, 30,
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

                # Converte o quadro para um blob e passe esse blob para a rede neural
                # e obtem a direção do movimento
                blob = cv2.dnn.blobFromImage(quadro, 0.007843, (W, H), 127.5)
                net.setInput(blob)
                deteccao = net.forward()

                # Roda o loop para as detecções
                for i in np.arange(0, deteccao.shape[2]):
                    # extrai o nível de confiança associada a previsão
                    confianca = deteccao[0, 0, i, 2]

                    # filtra detecções fracas exigindo um nível mínimo de confiança
                    if confianca > 0.4:
                        # extrai o índice das classes da lista de detecção
                        idx = int(deteccao[0, 0, i, 1])

                        # Se a classe não for uma pessoa ignore.
                        if CLASSES[idx] != "person":
                            continue

                        # calcula as coordenadas (x, y) da caixa de colisão para cada
                        # objeto

                        box = deteccao[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (sX, sY, eX, eY) = box.astype("int")

                        # constroi a abstração geométrica da correlação do dlib
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(sX, sY, eX, eY)
                        tracker.start_track(rgb, rect)

                        # adiciona o rastreador na lista de rastreadores
                        trackers.append(tracker)

            else:
                # loop para os rastreadores
                for tracker in trackers:
                    status = "Rastreando"

                    # atualiza o rastreador e armazena posição
                    tracker.update(rgb)
                    posicao = tracker.get_position()

                    # armazena posição do objeto
                    sX = int(posicao.left())
                    sY = int(posicao.top())
                    eX = int(posicao.right())
                    eY = int(posicao.bottom())

                    # adiciona posição do objeto para o array de colisões
                    rects.append((sX, sY, eX, eY))

            # desenha uma linha horizontal no centro do quadro, uma vez que um objeto cruza
            # a linha ele é determinado se ele esta entrando ou saindo
            cv2.line(quadro, (0, H // 2),
                     (W, H // 2), (0, 255, 0), 2)

            # Usar o rastreador do rastreador do centroid para associar o centroid antigo
            # com os centroids novos recém computados
            objects = ct.update(rects)

            # loop pelos objetos rastreados
            for (objID, centroid) in objects.items():
                # verifica que o objeto rastreado existe para o objeto atual
                to = trackableObjects.get(objID, None)

                # se não existe o objeto rastreado, crie um
                if to is None:
                    to = TrackableObj(objID, centroid)

                # caso contrário, existe um objeto rastreado que podemos usar para determinar a direção
                else:
                    # a diferença entre a coordernada y do centroid atual e do centroid
                    # anterior nos ajuda a determinar se o objeto esta se movendo para
                    # dentro do espaço público ou para fora

                    x = [i[1] for i in to.centroids]

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
                trackableObjects[objID] = to

                # desenher tanto o ID do objeto e o centroid no quadro de saída
                text = "ID {}".format(objID)
                cv2.putText(quadro, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(
                    quadro, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

            # constroi uma tupla com as informações mostradas no quadro
            info = [
                ("Entrando", totalUp),
                ("Saindo", totalDown),
                ("Lotacao", totalUp - totalDown),
            ]

            # itera sobre o conteúdo da tupla e desenha no quadro
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(quadro, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # verifique se o quadro deve ser gravado em disco
            if writer is not None:
                writer.write(quadro)

            # mostra o quadro de saída
            cv2.imshow("Frame", quadro)
            key = cv2.waitKey(1) & 0xFF

            # se a tecla `q` for pressionado quebre o loop
            if key == ord("q"):
                break

            # incrementa o número total de frames processados e atualiza o contador de FPS
            totalFrames += 1
            fps.update()

        # pare o timer e mostre o FPS
        fps.stop()

        # verifique se é necessário liberar o ponteiro de escrita
        if writer is not None:
            writer.release()

        # se um arquivo de vídeo não esta sendo utilizado pare o stream de vídeo
        if not self.argumentos.get("input", False):
            vs.stop()

        # caso contrário, libere o ponteiro de escrita do vídeo
        else:
            vs.release()

        # feche todas as janelas
        cv2.destroyAllWindows()

CrowdControl()