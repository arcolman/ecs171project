from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# ML models
models = {
    "Linear Regression": joblib.load("LinRegModel.joblib"),
    "Ridge Regression": joblib.load("RidgeRegModel.joblib"),
    "Random Forest": joblib.load("RandForestModel.joblib")
}


@app.route("/", methods=["GET", "POST"])
def index():

    results = None
    team_a = ""
    team_b = ""

    if request.method == "POST":

        # team names
        team_a = request.form["team_a_name"]
        team_b = request.form["team_b_name"]

        # team A stats
        a_fg = float(request.form["a_fg"])
        a_stl = float(request.form["a_stl"])
        a_x3p = float(request.form["a_x3p"])
        a_ast = float(request.form["a_ast"])
        a_ft = float(request.form["a_ft"])
        a_x2p = float(request.form["a_x2p"])

        # team B stats
        b_fg = float(request.form["b_fg"])
        b_stl = float(request.form["b_stl"])
        b_x3p = float(request.form["b_x3p"])
        b_ast = float(request.form["b_ast"])
        b_ft = float(request.form["b_ft"])
        b_x2p = float(request.form["b_x2p"])

        # compute diff in features (Team A - Team B)
        diff_in_features = [
            a_fg - b_fg,
            a_stl - b_stl,
            a_x3p - b_x3p,
            a_ast - b_ast,
            a_ft - b_ft,
            a_x2p - b_x2p
        ]

        # Convert to sklearn input shape
        X = np.array(diff_in_features).reshape(1, -1)

        results = []

        # get prediction from each model
        for name, model in models.items():
            prediction = model.predict(X)[0]
            # because team A-B so pos favors team A
            if prediction > 0:
                winner = team_a
            else:
                winner = team_b
            results.append({
                "model": name,
                "winner": winner
            })
    
    # render updates into index.html
    return render_template(
        "index.html",
        results=results,
        team_a=team_a,
        team_b=team_b
    )


if __name__ == "__main__":
    app.run(debug=True)