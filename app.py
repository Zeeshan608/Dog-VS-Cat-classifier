# Flask app: "/" shows the web page (front end), "/predict" runs the model (back end)
import os

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # fallback if the other package is installed instead
    from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)

# Load the model once when the app starts
MODEL_PATH = os.path.join(os.path.dirname(__file__), "dog_cat.tflite")
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
INPUT_INDEX = interpreter.get_input_details()[0]["index"]
OUTPUT_INDEX = interpreter.get_output_details()[0]["index"]
IMG_SIZE = 224  # same size used in the notebook

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dog or cat?</title>
  <style>
    :root {
      --bg: #eef2f6;
      --card: #ffffff;
      --ink: #1c2733;
      --muted: #5d6b79;
      --line: #c9d3dd;
      --accent: #2456d6;
      --dog: #c2571a;
      --cat: #2456d6;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #12171d;
        --card: #1b232c;
        --ink: #e8eef4;
        --muted: #9aa8b6;
        --line: #33414f;
        --accent: #7ea2ff;
        --dog: #f09a5f;
        --cat: #7ea2ff;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 32px 16px 48px;
      background: var(--bg); color: var(--ink);
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      line-height: 1.5;
    }
    main { max-width: 460px; margin: 0 auto; }
    h1 { font-size: 2.4rem; line-height: 1.1; margin: 0 0 8px; letter-spacing: -0.02em; }
    .lead { color: var(--muted); margin: 0 0 24px; }
    .card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 20px; }
    .drop {
      display: block; text-align: center; cursor: pointer;
      border: 2px dashed var(--line); border-radius: 10px; padding: 28px 12px;
      color: var(--muted);
    }
    .drop:hover, .drop.over, .drop:focus-within { border-color: var(--accent); color: var(--ink); }
    .drop input { position: absolute; opacity: 0; width: 1px; height: 1px; }
    #preview { display: none; width: 100%; max-height: 320px; object-fit: contain; margin-top: 16px; border-radius: 10px; }
    button {
      width: 100%; margin-top: 16px; padding: 12px; font: inherit; font-weight: 600;
      color: #fff; background: var(--accent); border: 0; border-radius: 10px; cursor: pointer;
    }
    @media (prefers-color-scheme: dark) { button { color: #0d1117; } }
    button:disabled { opacity: 0.55; cursor: default; }
    button:focus-visible, .drop:focus-within { outline: 3px solid var(--accent); outline-offset: 2px; }
    #result { margin-top: 20px; display: none; }
    #answer { font-size: 1.6rem; font-weight: 700; margin: 0 0 12px; }
    .row { display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 0.95rem; }
    .row span:first-child { width: 36px; }
    .row span:last-child { width: 52px; text-align: right; color: var(--muted); }
    .bar { flex: 1; height: 10px; background: var(--line); border-radius: 6px; overflow: hidden; }
    .fill { height: 100%; width: 0; transition: width 0.4s ease; }
    #dogFill { background: var(--dog); }
    #catFill { background: var(--cat); }
    #error { color: #c0392b; margin-top: 12px; display: none; }
    @media (prefers-reduced-motion: reduce) { .fill { transition: none; } }
  </style>
</head>
<body>
  <main>
    <h1>Dog or cat?</h1>
    <p class="lead">Upload a picture and the model will tell you which one it sees.</p>

    <div class="card">
      <label class="drop" id="drop">
        <input type="file" id="file" accept="image/*">
        <span id="dropText">Choose a picture or drop it here</span>
      </label>
      <img id="preview" alt="Your picture">
      <button id="go" disabled>Check picture</button>

      <div id="result" aria-live="polite">
        <p id="answer"></p>
        <div class="row"><span>Dog</span><div class="bar"><div class="fill" id="dogFill"></div></div><span id="dogPct"></span></div>
        <div class="row"><span>Cat</span><div class="bar"><div class="fill" id="catFill"></div></div><span id="catPct"></span></div>
      </div>
      <p id="error" role="alert"></p>
    </div>
  </main>

  <script>
    const fileInput = document.getElementById('file');
    const drop = document.getElementById('drop');
    const dropText = document.getElementById('dropText');
    const preview = document.getElementById('preview');
    const go = document.getElementById('go');
    const resultBox = document.getElementById('result');
    const answer = document.getElementById('answer');
    const errorBox = document.getElementById('error');

    function showPicture(file) {
      if (!file || !file.type.startsWith('image/')) {
        errorBox.textContent = 'This file is not a picture. Choose a JPG or PNG image.';
        errorBox.style.display = 'block';
        return;
      }
      errorBox.style.display = 'none';
      resultBox.style.display = 'none';
      preview.src = URL.createObjectURL(file);
      preview.style.display = 'block';
      dropText.textContent = file.name;
      go.disabled = false;
    }

    fileInput.addEventListener('change', () => showPicture(fileInput.files[0]));
    drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('over'));
    drop.addEventListener('drop', (e) => {
      e.preventDefault();
      drop.classList.remove('over');
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        showPicture(fileInput.files[0]);
      }
    });

    function setBars(dog) {
      const cat = 100 - dog;
      document.getElementById('dogFill').style.width = dog + '%';
      document.getElementById('catFill').style.width = cat + '%';
      document.getElementById('dogPct').textContent = dog.toFixed(1) + '%';
      document.getElementById('catPct').textContent = cat.toFixed(1) + '%';
    }

    go.addEventListener('click', () => {
      go.disabled = true;
      go.textContent = 'Checking…';
      errorBox.style.display = 'none';

      // Shrink the picture in the browser so the upload stays small
      const canvas = document.createElement('canvas');
      canvas.width = 224;
      canvas.height = 224;
      canvas.getContext('2d').drawImage(preview, 0, 0, 224, 224);

      canvas.toBlob(async (blob) => {
        const form = new FormData();
        form.append('file', blob, 'image.jpg');
        try {
          const res = await fetch('/predict', { method: 'POST', body: form });
          const data = await res.json();
          if (!res.ok) throw new Error(data.error || 'Request failed');
          answer.textContent = 'It looks like a ' + data.label.toLowerCase() + '.';
          setBars(data.dog_percent);
          resultBox.style.display = 'block';
        } catch (err) {
          errorBox.textContent = 'Could not check this picture. Try another image.';
          errorBox.style.display = 'block';
        }
        go.disabled = false;
        go.textContent = 'Check picture';
      }, 'image/jpeg', 0.95);
    });
  </script>
</body>
</html>
"""


@app.route("/")
def home():
    return PAGE


@app.route("/predict", methods=["POST"])
def predict():
    try:
        img = Image.open(request.files["file"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        # The model rescales pixels itself, so we send raw 0-255 values
        arr = np.expand_dims(np.array(img, dtype="float32"), axis=0)
        interpreter.set_tensor(INPUT_INDEX, arr)
        interpreter.invoke()
        prob_dog = float(interpreter.get_tensor(OUTPUT_INDEX)[0][0])  # 0 = cat, 1 = dog
        return jsonify(
            label="Dog" if prob_dog > 0.5 else "Cat",
            dog_percent=round(prob_dog * 100, 1),
        )
    except Exception:
        return jsonify(error="Could not read this picture."), 400


if __name__ == "__main__":
    app.run(debug=True)
