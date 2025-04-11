from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo
with open('adaboost_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        try:
            # Obtener datos del formulario
            gender = int(request.form["Gender"])
            fat = float(request.form["Fat_Percentage"])
            water = float(request.form["Water_Intake (liters)"])
            frequency = float(request.form["Workout_Frequency (days/week)"])
            experience = int(request.form["Experience_Level"])
            bmi = float(request.form["BMI"])          
            
            # Preparar datos
            features = np.array([[gender, fat, water, frequency, experience, bmi]])
            prediction = model.predict(features)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
