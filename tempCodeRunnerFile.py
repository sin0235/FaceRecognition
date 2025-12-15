import os
import sys

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

# ====== PATH SETUP ======
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from inference.recognition_engine import RecognitionEngine

# ====== FLASK APP ======
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

engine = RecognitionEngine()


# =====================================================
# TRANG CHỦ DUY NHẤT – ĐIỀU HƯỚNG 5 MENU
# =====================================================
@app.route("/", methods=["GET", "POST"])
def home():
    page = request.args.get("page", "recognition")

    # ====== BIẾN DÙNG CHUNG ======
    image_url = None
    result_name = None
    result_score = None

    # ====== NHẬN DIỆN ======
    if page == "recognition" and request.method == "POST":
        file = request.files.get("image")
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)

            img = Image.open(save_path).convert("RGB")
            img.save(save_path)

            name, score = engine.recognize(save_path)

            image_url = url_for("static", filename=f"uploads/{filename}")
            result_name = name
            result_score = f"{score:.4f}"

    # ====== SO SÁNH MODEL ======
    if page == "compare" and request.method == "POST":
        file = request.files.get("image")
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)

            img = Image.open(save_path).convert("RGB")
            img.save(save_path)

            # demo: dùng chung engine cho 2 model
            res_a = engine.recognize(save_path)
            res_b = engine.recognize(save_path)

            image_url = url_for("static", filename=f"uploads/{filename}")
            result_name = (res_a, res_b)

    # ====== THRESHOLD ======
    threshold_value = 0.5
    if page == "threshold" and request.method == "POST":
        threshold_value = float(request.form.get("threshold", 0.5))

        file = request.files.get("image")
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)

            img = Image.open(save_path).convert("RGB")
            img.save(save_path)

            name, score = engine.recognize(save_path)

            image_url = url_for("static", filename=f"uploads/{filename}")
            result_name = name
            result_score = score

    # ====== DATASET (DEMO) ======
    dataset_stats = {
        "num_people": 0,
        "num_images": 0,
        "avg_images": 0
    }

    # ====== DASHBOARD (DEMO) ======
    dashboard_stats = {
        "total": 0,
        "correct": 0,
        "accuracy": 0
    }

    return render_template(
        "home.html",
        page=page,
        image_url=image_url,
        result_name=result_name,
        result_score=result_score,
        threshold_value=threshold_value,
        dataset=dataset_stats,
        dashboard=dashboard_stats
    )


# =====================================================
if __name__ == "__main__":
    app.run(debug=True)
