import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image
from model.simple_cnn import SimpleCNN

# Initialize the Flask application
app = Flask(__name__)

# # Load the trained model
model = SimpleCNN(num_classes=7)  # because we have 7 classes in our dataset
model.load_state_dict(torch.load('./model/emotion_detection_model.pth', map_location=torch.device('cpu')))
model.eval()


# Define a dictionary to map label indices to emotion names
emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# Define transforms for incoming images
predict_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return render_template('/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        img = Image.open(file).convert('RGB')
        img_tensor = predict_transforms(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_emotion = emotion_dict[predicted.item()]

        return jsonify(predicted_emotion=predicted_emotion)

if __name__ == "__main__":
    app.run(debug=True)