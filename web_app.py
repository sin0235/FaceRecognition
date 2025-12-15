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
        "total": 100,
        "correct": 76,
        "wrong": 24,
        "accuracy": 76
    }
    return render_template("dashboard.html", active="dashboard", metrics=metrics)


if __name__ == "__main__":
    app.run(debug=True)
