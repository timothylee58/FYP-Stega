from flask import Flask, render_template, jsonify
import subprocess
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def run_subprocess(script):
    result = subprocess.run(["python", script], capture_output=True, text=True)
    return result.returncode, result.stderr  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/openEmbedTextWindow', methods=['POST'])
def open_embed_text_window():
    return_code, stderr = run_subprocess('VidStega - Embed text.py')
    if return_code == 0:
        return jsonify({'message': 'Text Embedded Successfully', 'success': True})
    else:
        return jsonify({'message': f'Operation Failed: {stderr}', 'success': False})

@app.route('/openExtractTextWindow', methods=['POST'])
def open_extract_text_window():
    return_code, stderr = run_subprocess('VidStega - Extract Hidden Text.py')
    if return_code == 0:
        return jsonify({'message': 'Extraction Successful', 'success': True})
    else:
        return jsonify({'message': f'Operation Failed: {stderr}', 'success': False})

if __name__ == '__main__':
    app.run(debug=True)