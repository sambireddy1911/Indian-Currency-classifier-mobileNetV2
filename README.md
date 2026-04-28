Indian Currency Classifier: MobileNetV2 with Test-Time Augmentation
This repository contains a complete deep learning pipeline for the automatic recognition of Indian currency denominations. The system uses a MobileNetV2 classifier with a transfer learning approach to identify 7 distinct Indian currency classes, trained on a dataset of approximately 6,000 images.

The trained network is exported to a 2.7MB TensorFlow Lite format for offline mobile deployment. You can integrate this model directly into Android applications for ATMs, cash counters, and assistive tools designed to help visually impaired users with independent money handling.

Key Results
The model was evaluated after training on the 6,000-image dataset containing varied capture conditions (lighting, angles, background noise, and note wear).

Validation Accuracy: 94.0%

Dataset Size: ~6,000 images

Number of Classes: 7 (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000)

Model Size: 2.7MB TFLite

Class-Wise Performance (F1-Score)
₹200: 0.99

₹2000: 0.98

₹500: 0.96

₹50: 0.91

₹100: 0.94

₹20: 0.94

₹10: 0.91

Android App Integration
You can deploy this model into a Kotlin-based Android application. The recommended implementation uses the CameraX API to handle image capture efficiently.

Do not run continuous frame-by-frame inference. You should configure the app to capture and analyze one frame at set intervals. This single-frame approach prevents device overheating and reduces battery usage. The lightweight 2.7MB TFLite file ensures the app works completely offline without network latency.

kotlin
// Load the 2.7MB TFLite model
val interpreter = InterpreterFactory().create(
    loadModelFile("currency_classifier.tflite"), 
    Interpreter.Options()
)

// Run inference on captured frame
val result = interpreter.run(cameraImage)
textView.text = "${result.className} (${result.confidence * 100}%)"
Technical Architecture
The architecture relies on MobileNetV2 pre-trained on ImageNet. The original top classification layers were removed, and custom dense layers were added to map the extracted features to the 7 output classes.

Network Flow:
Input Image -> Resize (224x224) & Normalize -> MobileNetV2 Base -> Custom Dense Layers -> Softmax(7)

Training & Optimization:

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Data Augmentation: Rotation, zoom, shift, and flip variations to simulate real-world input changes.

Test-Time Augmentation (TTA): Applied to improve prediction stability in uncertain cases, pushing borderline confidence scores from ~87% to over 95%.

Quickstart Guide
You can reproduce the training pipeline using Google Colab.

Open Indian_Currency_Colab.ipynb in Google Colab and enable the GPU runtime.

Mount your Google Drive to load the 6,000-image dataset.

Run all cells to train the model and generate the evaluation files.

Download the resulting currency_classifier.tflite file for your mobile project.

Repository Structure
text
Indian-Currency-Classifier-MobileNetV2/
├── Indian_Currency_Colab.ipynb      # Complete training pipeline
├── currency_classifier.tflite       # Production-ready 2.7MB model
├── class_names.txt                  # List of 7 denomination classes
├── LICENSE                          # Academic view-only license
├── .gitignore
└── screenshots/
    ├── currency_classifier_workflow_v2.png
    ├── confusion_matrix.png
    └── demo.png
Academic Citation
If you use this project in your research, please cite it using the following format:

text
@misc{sambi2026indian,
  title = {Indian Currency Classifier: MobileNetV2 with Test-Time Augmentation},
  author = {ARE SAMBI REDDY},
  year = {2026},
  month = {April},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/sambi/Indian-Currency-Classifier-MobileNetV2}},
  note = {94.0\% validation accuracy, 6000 images, 7 classes}
}
Developer: ARE SAMBI REDDY
Registration Number: 12205687
Location: Lovely Professional University, Punjab
Contact: sambi1911329@gmail.com
