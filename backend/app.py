from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from tp.video_indexing import getInit

app = Flask(__name__)
CORS(app)

LOCAL_VIDEO_FOLDER = "./tp/compressedVideos"
OUTPUT_VIDEO = "output.mp4"

@app.route('/process_sentence', methods=['POST'])
def process_sentence_api():
    """
    API to process a sentence and generate a corresponding video.
    """
    data = request.get_json()
    if not data or 'sentence' not in data:
        return jsonify({"error": "Missing 'sentence' in request body"}), 400

    sentence = data['sentence']
    print(f"Processing sentence: {sentence}")

    result = getInit(sentence)
    if not result:
        return jsonify({"message": "No matching videos found for the given sentence."}), 404

    return send_file(OUTPUT_VIDEO, mimetype="video/mp4", as_attachment=True, download_name="output.mp4")

if __name__ == '__main__':
    app.run(debug=True, port=5000)