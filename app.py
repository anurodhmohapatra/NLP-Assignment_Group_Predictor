import pickle

from flask import Flask, render_template, request

# load the model from disk
clf = pickle.load(open('clf.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))
le = pickle.load(open('le.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['Description']
        data = [message]
        my_prediction = predict(data)
    return render_template('result.html', prediction=my_prediction)


def predict(description):
    prediction = clf.predict(cv.transform(description))
    return le.inverse_transform(prediction)[0]


if __name__ == '__main__':
    app.run(debug=True)
