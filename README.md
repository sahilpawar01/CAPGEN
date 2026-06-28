# Automatic Image Captioning System using CNN-LSTM (Deep Learning)

## Overview

The **Automatic Image Captioning System** is a deep learning-based application that automatically generates natural language descriptions for images. The project combines a **Convolutional Neural Network (CNN)** for extracting visual features with a **Long Short-Term Memory (LSTM)** network for generating meaningful captions.

This project was developed as a **Final Year Engineering Project** to demonstrate the application of Computer Vision, Natural Language Processing (NLP), and Deep Learning in image understanding.

---

## Features

* Upload an image and generate descriptive captions.
* CNN-based feature extraction using **InceptionV3**.
* LSTM-based sequence generation for captions.
* Tokenization and word embedding for text generation.
* Fast caption generation using a trained deep learning model.
* Clean and responsive web interface.
* Docker support for easy deployment.

---

## Technologies Used

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras

### Computer Vision

* OpenCV
* InceptionV3

### Natural Language Processing

* LSTM
* Tokenizer
* Word Embeddings

### Web Technologies

* Flask
* HTML
* CSS
* JavaScript

### Deployment

* Docker

---

## Dataset

The model is trained on the **Flickr8k Dataset**, which contains:

* 8,000 images
* 5 captions per image
* More than 40,000 human-written captions

---

## Model Architecture

```text
Image
   │
   ▼
InceptionV3 (CNN)
   │
2048-D Feature Vector
   │
   ▼
Dense Layer
   │
   ▼
LSTM Decoder
   │
   ▼
Generated Caption
```

---

## Project Structure

```text
Automatic-Image-Captioning/
│
├── static/
├── templates/
├── uploads/
├── models/
├── app.py
├── model.py
├── feature_extraction.py
├── requirements.txt
├── Dockerfile
├── README.md
└── ...
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/sahilpawar01/CAPGEN.git

cd CAPGEN
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

```bash
python app.py
```

Open your browser and navigate to:

```text
http://127.0.0.1:5000
```

---

## Docker

### Build the Docker Image

```bash
docker build -t image-captioning .
```

### Run the Docker Container

```bash
docker run -p 5000:5000 image-captioning
```

---

## How It Works

1. The user uploads an image.
2. InceptionV3 extracts high-level visual features.
3. The extracted feature vector is passed to the LSTM decoder.
4. The LSTM predicts the caption one word at a time.
5. The final caption is displayed on the web interface.

---

## Future Enhancements

* Integration of BLIP and BLIP-2 models
* Beam Search for improved caption generation
* Object detection using YOLOv8
* Face recognition support
* Real-time webcam captioning
* Multilingual caption generation
* Voice output for visually impaired users

---

## Sample Output

```text
Input:
Image of a dog running in a grassy field

Generated Caption:
"A brown dog is running through the grass."
```

---

## Applications

* Assistive technology for visually impaired users
* Image indexing and retrieval
* Smart photo organization
* Robotics
* Surveillance systems
* Social media content generation

---

## Author

**Sahil Pawar**

Bachelor of Engineering (Computer Engineering)

Final Year Project

---

## License

This project is intended for educational and research purposes.
