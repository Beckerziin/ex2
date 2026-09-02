"""
Atividade Prática - Detecção de Objetos em Vídeo
Centro Universitário FAG

Detecta e classifica objetos em um vídeo (arquivo local ou URL do YouTube)
usando YOLO (ultralytics) + OpenCV. Desenha as caixas delimitadoras
(bounding boxes) e as classes durante a execução do vídeo.

Uso:
    python main.py                                   # usa o vídeo padrão (webcam 0)
    python main.py --source video.mp4                # arquivo de vídeo local
    python main.py --source "https://youtu.be/XXXX"  # URL do YouTube
    python main.py --source 0                        # webcam

Opções úteis:
    --model yolov8n.pt      modelo YOLO (baixado automaticamente na 1ª vez)
    --conf 0.35             confiança mínima para exibir a detecção
    --save saida.mp4        grava o resultado em um arquivo de vídeo
    --classes person car    filtra apenas as classes informadas

Pressione 'q' na janela do vídeo para encerrar.
"""

import argparse
import sys
import time

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit(
        "Pacote 'ultralytics' não encontrado.\n"
        "Instale as dependências com:\n"
        "    python -m pip install ultralytics opencv-python yt-dlp"
    )


def resolve_youtube_url(url: str) -> str:
    """Converte uma URL do YouTube em uma URL de stream direta usando yt-dlp."""
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        sys.exit(
            "Pacote 'yt-dlp' não encontrado (necessário para vídeos do YouTube).\n"
            "Instale com: python -m pip install yt-dlp"
        )

    ydl_opts = {
        # prioriza um mp4 progressivo até 720p (stream único, fácil de ler no OpenCV)
        "format": "best[ext=mp4][height<=720]/best[height<=720]/best",
        "quiet": True,
        "no_warnings": True,
    }
    print(f"[yt-dlp] Resolvendo stream de: {url}")
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if "url" in info:
            return info["url"]
        # alguns formatos vêm aninhados
        for fmt in reversed(info.get("formats", [])):
            if fmt.get("url"):
                return fmt["url"]
    sys.exit("Não foi possível obter o stream do vídeo do YouTube.")


def is_youtube(source: str) -> bool:
    s = source.lower()
    return "youtube.com" in s or "youtu.be" in s


def open_capture(source: str) -> cv2.VideoCapture:
    """Abre o VideoCapture a partir de webcam (índice), arquivo local ou YouTube."""
    if source.isdigit():
        return cv2.VideoCapture(int(source), cv2.CAP_DSHOW if sys.platform == "win32" else 0)
    if is_youtube(source):
        source = resolve_youtube_url(source)
    return cv2.VideoCapture(source)


def color_for_class(class_id: int):
    """Gera uma cor BGR estável e bem distinta para cada id de classe."""
    hue = (class_id * 37) % 180  # espalha o matiz pelo círculo HSV
    hsv = np.uint8([[[hue, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def draw_detection(frame, box, label, color):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Detecção de objetos em vídeo (YOLO + OpenCV)")
    parser.add_argument("--source", default="0",
                        help="Arquivo de vídeo local, URL do YouTube ou índice da webcam (padrão: 0)")
    parser.add_argument("--model", default="yolov8n.pt", help="Modelo YOLO (padrão: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confiança mínima (padrão: 0.35)")
    parser.add_argument("--save", default=None, help="Caminho para gravar o vídeo com as detecções")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Filtra apenas estas classes (ex.: person car bus)")
    parser.add_argument("--no-window", action="store_true",
                        help="Não exibe a janela (útil ao apenas gravar com --save)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[YOLO] Carregando modelo: {args.model}")
    model = YOLO(args.model)
    names = model.names  # {id: 'nome'}

    # converte nomes de classe -> ids para o filtro
    class_filter = None
    if args.classes:
        wanted = {c.lower() for c in args.classes}
        class_filter = [i for i, n in names.items() if n.lower() in wanted]
        if not class_filter:
            sys.exit(f"Nenhuma classe válida em {args.classes}. Classes disponíveis: {sorted(names.values())}")
        print(f"[YOLO] Filtrando classes: {[names[i] for i in class_filter]}")

    cap = open_capture(args.source)
    if not cap.isOpened():
        sys.exit(f"Não foi possível abrir a fonte de vídeo: {args.source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (width, height))
        print(f"[OpenCV] Gravando em: {args.save}")

    window = "Deteccao de Objetos - FAG (q para sair)"
    if not args.no_window:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_count = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        results = model.predict(
            frame, conf=args.conf, classes=class_filter, verbose=False
        )[0]

        counts = {}
        for b in results.boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            name = names.get(cls_id, str(cls_id))
            counts[name] = counts.get(name, 0) + 1
            draw_detection(frame, (x1, y1, x2, y2), f"{name} {conf:.2f}", color_for_class(cls_id))

        # HUD: fps e contagem por classe
        fps = frame_count / (time.time() - t0) if time.time() > t0 else 0.0
        resumo = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "nenhum objeto"
        cv2.putText(frame, f"FPS: {fps:.1f} | {resumo}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)

        if not args.no_window:
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Encerrado pelo usuário.")
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Fim. {frame_count} frames processados em {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
