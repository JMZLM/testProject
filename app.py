from flask import Flask, render_template, request
from models.bug_localizer import BugLocalizer

app = Flask(__name__)
localizer = BugLocalizer()

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    if request.method == "POST":
        code   = request.form["code"]
        report = request.form["report"]
        scored = localizer.predict(code, report)
        top10  = sorted(scored, key=lambda x: x[1], reverse=True)[:10]
        results = [
            {"line_no": ln, "score": f"{sc:.3f}", "text": code.splitlines()[ln-1]}
            for ln, sc in top10
        ]
    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
