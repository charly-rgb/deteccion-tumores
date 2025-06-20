import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
import traceback
from werkzeug.utils import secure_filename
from processing.tumor_detector import TumorDetector
from processing.image_processor import generate_visualizations

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Crear directorios si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict_tumor():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file provided'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        detector = TumorDetector()
        result = detector.detect_tumor(filepath)

        # Asegúrate que los valores son serializables
        has_tumor_val = bool(result.get('has_tumor'))
        confidence_val = float(result.get('confidence', 0.0))

        return jsonify({
            'has_tumor': has_tumor_val,
            'confidence': f"{confidence_val:.2f}%",
            'filename': filename
        })

    except Exception as e:
        import traceback
        print("[ERROR]", traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Procesamiento de la imagen
            detector = TumorDetector()
            result = detector.detect_tumor(filepath)
            
            # Generar visualizaciones
            base_name = os.path.splitext(filename)[0]
            result_images = generate_visualizations(
                filepath, 
                app.config['RESULT_FOLDER'], 
                base_name
            )
            
            original_image_path = os.path.basename(filepath)  # Solo el nombre del archivo
            return render_template('index.html',
                                original_image=original_image_path,
                                result_images=result_images,
                                has_tumor=result['has_tumor'],
                                confidence=f"{result['confidence']:.2f}%")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
