# Installing EcoMotionZip — Desktop (Mac / Windows)

This guide is for users who want the graphical desktop interface.

---

## What you need

- A Mac (macOS 11+) or Windows (10/11) computer
- Python 3.8 or newer — download from [python.org](https://www.python.org/downloads/)
- About 500 MB of free disk space

---

## Step 1 — Download EcoMotionZip

Go to the [Releases page](https://github.com/malika_ratnayake/EcoMotionZip/releases) and download
the latest `.zip` file. Extract it anywhere you like (e.g. your Desktop or Documents folder).

> If you are comfortable with Git, you can also clone the repository:
> `git clone https://github.com/malika_ratnayake/EcoMotionZip.git`

---

## Step 2 — Open a terminal in the EcoMotionZip folder

**Mac:** Right-click the extracted folder → "New Terminal at Folder"
*(or open Terminal and drag the folder onto it)*

**Windows:** Open the folder → click the address bar → type `cmd` → press Enter

---

## Step 3 — Create a virtual environment (recommended)

This keeps EcoMotionZip's dependencies separate from other Python software.

```bash
python -m venv emz_venv
```

Then activate it:

**Mac / Linux:**
```bash
source emz_venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

You should see `(emz_venv)` at the start of your command prompt.

---

## Step 4 — Install dependencies

```bash
pip install -r requirements-gui.txt
```

This installs OpenCV, NumPy, and the PySide6 graphical toolkit.
It may take a few minutes the first time.

---

## Step 5 — Run the application

```bash
python run_desktop.py
```

The EcoMotionZip window will open. You can now select your video files,
adjust settings, and start processing.

---

## Updating EcoMotionZip

Download the latest release zip, extract it over your existing folder,
then run `pip install -r requirements-gui.txt` again to pick up any
new dependencies.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python: command not found` | Try `python3` instead of `python` |
| `pip: command not found` | Try `pip3` or `python -m pip` |
| Window does not open | Make sure you activated the virtual environment |
| Video not loading | Check the file path has no special characters |

For further help, open an issue at the [GitHub repository](https://github.com/malika_ratnayake/EcoMotionZip/issues).
