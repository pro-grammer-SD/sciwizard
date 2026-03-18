# Installation

## Requirements

| Dependency | Minimum version |
|------------|----------------|
| Python | 3.10 |
| PySide6 | 6.6.0 |
| scikit-learn | 1.4.0 |
| pandas | 2.1.0 |
| numpy | 1.26.0 |
| matplotlib | 3.8.0 |
| joblib | 1.3.0 |

---

## Install from source (recommended)

```bash
git clone https://github.com/pro-grammer-SD/sciwizard.git
cd sciwizard

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

The `-e` flag installs in editable mode — changes to the source are reflected immediately without reinstalling.

---

## Platform notes

### Windows

No extra steps. The application sets a Windows AppUserModelID automatically so the taskbar icon displays correctly.

Place your application icon at `icon/icon.ico` relative to the project root. If the file is absent, the app launches without an icon and logs a warning.

### macOS

PySide6 on macOS requires macOS 11+. No additional system packages needed.

### Linux

Install Qt platform dependencies before running:

```bash
# Debian / Ubuntu
sudo apt-get install libglib2.0-0 libgl1 libxcb-cursor0 libxkbcommon-x11-0

# Fedora / RHEL
sudo dnf install mesa-libGL libxkbcommon-x11
```

To run in a headless CI environment:

```bash
QT_QPA_PLATFORM=offscreen python -m sciwizard
```

---

## Verify the install

```bash
python -c "import sciwizard; print(sciwizard.__version__)"
python -m sciwizard --help   # not yet implemented — launches the GUI
```

---

## Uninstall

```bash
pip uninstall sciwizard
```

User data (saved models, experiment log) lives in `~/.sciwizard/` and is not removed by pip. Delete that directory manually if needed.
