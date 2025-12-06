import os
import sys

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from inference.recognition_engine import RecognitionEngine


app = Flask(__name__)

# thư mục lưu ảnh upload để show lên web
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

engine = RecognitionEngine()


@app.route("/", methods=["GET", "POST"])
def index():
    result_name = None
    result_score = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # đảm bảo ảnh lưu chuẩn
        img = Image.open(save_path).convert("RGB")
        img.save(save_path)

        # GỌI ĐÚNG HÀM recognize, KHÔNG PHẢI predict
        name, score = engine.recognize(save_path)

        result_name = name
        result_score = f"{score:.4f}"
        image_url = url_for("static", filename=f"uploads/{filename}")

    return render_template(
        "index.html",
        result_name=result_name,
        result_score=result_score,
        image_url=image_url,
    )


if __name__ == "__main__":
    app.run(debug=True)
