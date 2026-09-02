# Atividade Prática – Detecção de Objetos em Vídeo

Centro Universitário FAG

Algoritmo em Python que **identifica e classifica objetos** presentes em um vídeo
(arquivo local **ou** URL do YouTube), exibindo as **caixas delimitadoras
(bounding boxes)** e as **classes** durante a execução do vídeo.

Utiliza o modelo **YOLO** (via `ultralytics`) treinado no dataset COCO, que
reconhece 80 classes — entre elas pessoas, carros, motocicletas, ônibus,
bicicletas, caminhões e diversos animais.

## Instalação

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

(ou, conforme o enunciado: `python -m pip install ultralytics opencv-python yt-dlp`)

## Execução

```bat
:: vídeo local
python main.py --source video.mp4

:: vídeo do YouTube
python main.py --source "https://www.youtube.com/watch?v=XXXXXXXXXXX"

:: webcam
python main.py --source 0
```

Pressione **`q`** na janela do vídeo para encerrar.

## Opções

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `--source` | Arquivo local, URL do YouTube ou índice da webcam | `0` |
| `--model` | Modelo YOLO (baixado automaticamente na 1ª vez) | `yolov8n.pt` |
| `--conf` | Confiança mínima para exibir a detecção | `0.35` |
| `--save` | Grava o vídeo resultante em um arquivo | – |
| `--classes` | Filtra apenas as classes informadas (ex.: `person car bus`) | todas |
| `--no-window` | Não abre a janela (útil ao apenas gravar com `--save`) | – |

Exemplos:

```bat
:: só pessoas e veículos, gravando o resultado
python main.py --source transito.mp4 --classes person car motorcycle bus truck --save saida.mp4

:: modelo maior (mais preciso, mais lento)
python main.py --source video.mp4 --model yolov8s.pt
```

## Como funciona

1. `main.py` abre a fonte de vídeo com o OpenCV. Se for URL do YouTube, o
   `yt-dlp` resolve o endereço de stream direto (mp4 até 720p).
2. Para cada frame, o YOLO faz a inferência (`model.predict`).
3. Cada detecção vira uma bounding box com rótulo `classe + confiança`.
4. Um HUD no topo mostra o FPS e a contagem de objetos por classe no frame.
5. Com `--save`, cada frame anotado é gravado em vídeo.
