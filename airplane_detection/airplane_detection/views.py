import base64
import numpy as np
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.http import HttpRequest

import cv2


@require_http_methods(["GET", "POST"])
def index(request: HttpRequest):
    """
    Ein Formular:
      - <input type="file" name="ap-image-upload">
      - Button "Start Process" => verarbeitet direkt in-memory (ohne Speichern)
    Das Template zeigt den Vergleichsblock nur, wenn front_image_url & back_image_url gesetzt sind.
    """
    context = {}

    if request.method == "POST" and "process-image" in request.POST:
        uploaded_image = request.FILES.get("ap-image-upload")
        
        if not uploaded_image:
            context["error"] = "Bitte zuerst eine Bilddatei auswählen."
        else:
            # Speichert Bild in Bytes
            image_bytes = uploaded_image.read()
            # Gibt Link für Bild zurück, welcher Komplettes Bild ohne speichern anzeigt
            front_image_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("utf-8")

            # Umwandlung für Bildbearbeitung
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img is None:
                context["error"] = "Das hochgeladene Bild konnte nicht gelesen werden."
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                process_status, image_data = cv2.imencode(".png", gray)

                if not process_status:
                    context["error"] = "Verarbeitung fehlgeschlagen."
                else:
                    back_image_url = "data:image/png;base64," + base64.b64encode(image_data.tobytes()).decode("utf-8")
                    context["front_image_url"] = front_image_url
                    context["back_image_url"] = back_image_url

    return render(request, "index.html", context)
