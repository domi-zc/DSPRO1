# DSPRO1
Data Science Project I

## Frontend

A lightweight Django project without a database.
It serves as a **one-pager** to upload, process, and immediately display images.
**Nothing is stored.**

---

### Requirements
- **Python 3.13.0** (recommended via [pyenv](https://github.com/pyenv/pyenv))
- **Django 5.2.8**

---

### Installation

#### Linux & macOS

```bash
# Use pyenv
pyenv install -s 3.13.0
pyenv local 3.13.0

# Check version
python -V  # should display "Python 3.13.0"

# Create & activate virtual environment
python -m venv .dspro1_env
source .dspro1_env/bin/activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Start server
python airplane_detection/manage.py runserver
```

---

#### Windows (PowerShell)

```bash
# Ensure Python 3.13.0 is installed
# (download and install manually from [https://www.python.org/downloads/](https://www.python.org/downloads/))

# (optionally with pyenv-win)
# pyenv install 3.13.0
# pyenv local 3.13.0

# Check Python version
python -V  # should display "Python 3.13.0"

# Create & activate virtual environment
python -m venv .dspro1_env
.\.dspro1_env\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Start server
python airplane_detection\manage.py runserver
```