from flask import Flask, request, render_template, url_for
from sentiment import nlp

app = Flask(__name__)

@app.route("/home")
def home():
    return render_template("home.html")
'''
@app.route("/result",methods=["GET","POST"])
def output():
    form_data = request.form
    status = credit_card(form_data["card"])
    return render_template("result.html",status=status)
'''
@app.route("/result", methods=["GET","POST"])
def form_data():
        text = request.form['text']
        status = nlp(text)
        return render_template("result.html", status=status)


@app.route("/ourteam")
def team():
    return render_template("ourteam.html")

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
