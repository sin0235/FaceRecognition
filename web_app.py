from flask import Flask, render_template, request
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    image_name = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            image_name = file.filename
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], image_name))
            result = "Mỹ Tâm (0.76)"

    return render_template("home.html", active="home",
                           result=result, image_name=image_name)


@app.route("/compare", methods=["GET", "POST"])
def compare():
    result = None
    image_name = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            image_name = file.filename
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], image_name))
            result = {
                "a": "Mỹ Tâm (0.76)",
                "b": "Mỹ Tâm (0.71)"
            }

    return render_template("compare.html", active="compare",
                           result=result, image_name=image_name)


@app.route("/threshold", methods=["GET", "POST"])
def threshold():
    result = None
    image_name = None
    threshold_val = 0.5

    if request.method == "POST":
        threshold_val = request.form.get("threshold", 0.5)
        file = request.files.get("image")
        if file and file.filename:
            image_name = file.filename
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], image_name))
            result = f"Mỹ Tâm (MATCH)"

    return render_template("threshold.html", active="threshold",
                           result=result,
                           threshold=threshold_val,
                           image_name=image_name)


@app.route("/dataset")
def dataset():
    return render_template("dataset.html", active="dataset")


@app.route("/dashboard")
def dashboard():
    metrics = {
        "total": 200,
        "correct": 152,
        "wrong": 48,
        "accuracy": 76,
        "precision": 0.78,
        "recall": 0.75
    }

    return render_template(
        "dashboard.html",
        active="dashboard",
        metrics=metrics
    )

@app.route("/batch", methods=["GET", "POST"])
def batch():
    results = []

    if request.method == "POST":
        files = request.files.getlist("images")

        for file in files:
            if file and file.filename:
                save_path = os.path.join(
                    app.config["UPLOAD_FOLDER"], file.filename
                )
                file.save(save_path)

                # DEMO kết quả (sau này thay bằng ArcFace thật)
                results.append({
                    "image": file.filename,
                    "name": "My Tam",
                    "score": 0.76
                })

    return render_template(
        "batch.html",
        active="batch",
        results=results
    )

@app.route("/explain", methods=["GET", "POST"])
def explain():
    image_name = None
    gradcam_image = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            image_name = file.filename
            save_path = os.path.join(
                app.config["UPLOAD_FOLDER"], image_name
            )
            file.save(save_path)

            # Demo Grad-CAM (sau này thay bằng output thật)
            gradcam_image = "gradcam/demo_gradcam.jpg"

    return render_template(
        "explain.html",
        active="explain",
        image_name=image_name,
        gradcam_image=gradcam_image
    )
@app.route("/evaluation")
def evaluation():
    return render_template("evaluation.html", active="evaluation")



if __name__ == "__main__":
    app.run(debug=True)
