from flask import Flask,render_template
import pickle
app = Flask (__name__)
pickle_in = open("model.pkl","rb")
model = pickle.load(pickle_in)

@app.route("/")
def hello ():
    return "Welcome to my application "

app.route("/predict")
def predict():
    return render_template("index.html")
if __name__ == "__main__":
    app.run()