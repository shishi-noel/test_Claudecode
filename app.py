"""
離職予測 Web アプリ
Run: python app.py
"""
from flask import Flask, render_template, request
from turnover_model import predict, load_model

app = Flask(__name__)
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    form_data = {}

    if request.method == "POST":
        try:
            form_data = {
                "age":          int(float(request.form["age"])),
                "gender":       int(float(request.form["gender"])),
                "tenure":       int(float(request.form["tenure"])),
                "night_shifts": int(float(request.form["night_shifts"])),
                "stress":       float(request.form["stress"]),
            }

            prob = predict(**form_data, model=get_model())
            pct = round(prob * 100, 1)

            tenure = form_data["tenure"]
            if tenure >= 20:
                high_thresh, mid_thresh = 0.40, 0.20
            elif tenure >= 10:
                high_thresh, mid_thresh = 0.50, 0.25
            elif tenure >= 3:
                high_thresh, mid_thresh = 0.60, 0.35
            else:
                high_thresh, mid_thresh = 0.70, 0.40

            if prob >= high_thresh:
                risk_level = "high"
                risk_label = "高リスク"
            elif prob >= mid_thresh:
                risk_level = "medium"
                risk_label = "中リスク"
            else:
                risk_level = "low"
                risk_label = "低リスク"

            result = {"prob": pct, "risk_level": risk_level, "risk_label": risk_label}

        except (ValueError, KeyError) as e:
            error = str(e)

    return render_template("index.html", result=result, error=error, form_data=form_data)


if __name__ == "__main__":
    app.run(debug=True)
