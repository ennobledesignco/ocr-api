from flask import Flask, request, jsonify
import cv2
import pytesseract
import tempfile
import os

app = Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "online"})

@app.route("/ocr", methods=["POST"])
def ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    img_file = request.files['image']
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img_path = temp.name
    img_file.save(img_path)

    try:
        # Read image with OpenCV
        image = cv2.imread(img_path)

        # Preprocess
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, h=10)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Save processed temp image
        processed_path = img_path.replace(".png", "_processed.png")
        cv2.imwrite(processed_path, thresh)

        # Run Tesseract OCR (Spanish)
        text = pytesseract.image_to_string(processed_path, lang='spa')

        return jsonify({"text": text.strip()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        os.remove(img_path)
        if os.path.exists(processed_path):
            os.remove(processed_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
