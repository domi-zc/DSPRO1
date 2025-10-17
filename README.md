# DSPRO1
Data Science Projekt I

## Frontend

Ein schlankes Django-Projekt ohne Datenbank.  
Es dient als **One-Pager**, um Bilder hochzuladen, zu verarbeiten und direkt wieder anzuzeigen.
**Nichts wird gespeichert.**

---

### Anforderungen
- **Python 3.13.0** (empfohlen über [pyenv](https://github.com/pyenv/pyenv))
- **Django 5.2.7**

---

### Installation

#### Linux & macOS

```bash
# pyenv verwenden
pyenv install -s 3.13.0
pyenv local 3.13.0

# Version prüfen
python -V  # sollte "Python 3.13.0" anzeigen

# Virtuelle Umgebung anlegen & aktivieren
python -m venv django-ad-venv
source django-ad-venv/bin/activate

# Dependencies installieren
python -m pip install --upgrade pip
pip install -r requirements.txt

# Server starten
python airplane_detection/manage.py runserver
```

---

#### Windows (PowerShell)

```bash
# Stelle sicher, dass Python 3.13.0 installiert ist
# (manuell von https://www.python.org/downloads/ herunterladen und installieren)

# (optional mit pyenv-win)
# pyenv install 3.13.0
# pyenv local 3.13.0

# Python-Version prüfen
python -V  # sollte "Python 3.13.0" anzeigen

# Virtuelle Umgebung erstellen
python -m venv django-ad-venv

# Aktivieren
.\django-ad-venv\Scripts\Activate.ps1

# Abhängigkeiten installieren
python -m pip install --upgrade pip
pip install -r requirements.txt

# Server starten
python airplane_detection/manage.py runserver
```