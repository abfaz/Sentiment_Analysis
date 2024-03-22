from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('naive_bayes.pkl')

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/prediction', methods=['POST'])
def prediction():
    
    if request.method == 'POST':
        # Get the review from the form
        review = request.form.get('reviews')

        # Validate input
        if not review:
            return render_template('error.html', message='Please enter a review.')


        # Make prediction
        prediction = model.predict([review])

        # Determine sentiment based on prediction
        result = "👍 Positive Review 👍" if prediction[0] == 1 else "👎 Negative Review 👎"

        return render_template('output.html', prediction=result)
    else:
        return 'Method Not Allowed'

if __name__ == '__main__':
    app.run(debug=True)


    





if __name__ == "__main__":
    app.run(debug=True)