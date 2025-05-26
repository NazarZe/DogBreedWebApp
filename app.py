from flask import Flask, request, render_template, url_for, send_file, redirect
from main import predict_all_models, predict_mat_model, class_names, generate_plots
import os
import sqlite3
from collections import Counter

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

def get_care_tip(breed):
    conn = sqlite3.connect('care.db')
    cursor = conn.cursor()
    cursor.execute("SELECT advice FROM breed_care WHERE LOWER(breed) = ?", (breed.lower(),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return row[0]
    return "Поради для цієї породи поки що відсутні."

@app.route('/', methods=['GET', 'POST'])
def index():
    result = []
    image_path = None
    model_option = 'all'
    care_tips = None

    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            model_option = request.form.get('model_option')
            if file:
                filename = file.filename
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                image_url = url_for('static', filename='uploads/' + filename)

                if model_option == 'all':
                    result = predict_all_models(save_path, class_names)
                    result.append(("Stanford-MAT", "ExtraTrees", predict_mat_model(save_path)))
                elif model_option == 'Stanford-MAT':
                    result.append(("Stanford-MAT", "ExtraTrees", predict_mat_model(save_path)))
                else:
                    result = predict_all_models(save_path, class_names, model_filter=model_option)

                breeds = [pred.split('-')[-1] if '-' in pred else pred for _, _, pred in result if isinstance(pred, str)]
                if breeds:
                    most_common = Counter(breeds).most_common(1)[0][0]
                    care_text = get_care_tip(most_common)
                    care_tips = {"breed": most_common, "text": care_text}

                image_path = image_url

    return render_template('index.html', result=result, image=image_path,
                           selected_model=model_option, care_tips=care_tips)

@app.route('/generate-plots')
def generate_plots_route():
    import pandas as pd
    if os.path.exists("model_comparison_results.csv"):
        df = pd.read_csv("model_comparison_results.csv")
        generate_plots(df)
        return redirect(url_for('static', filename='accuracy_comparison.png'))
    return "Файл з результатами не знайдено."

@app.route('/retrain')
def retrain_model():
    return "Функціонал перенавчання ще не реалізовано."

if __name__ == '__main__':
    app.run(debug=True)
