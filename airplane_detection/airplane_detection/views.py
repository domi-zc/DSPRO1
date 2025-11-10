from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.http import HttpRequest
from django.conf import settings

import base64
import numpy as np
import cv2
import os.path

from ultralytics import YOLO

MODEL = YOLO(str(settings.MODEL_PATH))

@require_http_methods(["GET", "POST"])
def index(request: HttpRequest):
    context = {}

    if request.method == "POST" and "process-image" in request.POST:
        uploaded_image = request.FILES.get("ap-image-upload")
        if not uploaded_image:
            context["error"] = "Bitte zuerst eine Bilddatei auswählen."
        else:
            # Original als Data-URL zurückgeben
            content_type = getattr(uploaded_image, "content_type", "image/jpeg")
            image_bytes = uploaded_image.read()
            front_image_url = f"data:{content_type};base64," + base64.b64encode(image_bytes).decode("utf-8")

            # Bildnamen generieren falls heruntergeladen wird
            original_name = uploaded_image.name
            original_stem = os.path.splitext(original_name)[0]
            download_filename = f"{original_stem}-processed.png"

            # OpenCV-Decoding
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                context["error"] = "Das hochgeladene Bild konnte nicht gelesen werden."
            else:
                # YOLO-Inferenz (RGB)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                r = MODEL(img_rgb)[0]

                # Annotiertes Bild erzeugen
                annotated_rgb = r.plot()
                annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                ok, png_buf = cv2.imencode(".png", annotated_bgr)
                if not ok:
                    context["error"] = "Verarbeitung fehlgeschlagen."
                else:
                    back_image_url = "data:image/png;base64," + base64.b64encode(png_buf.tobytes()).decode("utf-8")

                    # Anzahl Flugzeuge zählen
                    names = getattr(r, "names", {})
                    plane_cls = next((k for k, v in names.items() if str(v).lower() == "plane"), None)
                    if plane_cls is None:
                        plane_count = len(r.boxes) if r.boxes is not None else 0
                    else:
                        plane_count = sum(1 for b in (r.boxes or []) if int(b.cls[0]) == int(plane_cls))

                    context.update({
                        "front_image_url": front_image_url,
                        "back_image_url": back_image_url,
                        "plane_count": plane_count,
                        "download_filename": download_filename,
                    })

    return render(request, "index.html", context)
