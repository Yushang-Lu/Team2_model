# Team2_model

A lightweight 10-layer CNN image classifier built with TensorFlow for 96×96 RGB images across 15 categories.

It can be trained on your computer and exported to ONNX for deployment on any ONNX‑compatible platform (e.g., mobile, web, edge devices).

Follow the steps below to get started! 🚀

---

## 1. 📁 Project Structure

```txt
Team2_model/
├── data/                    # Stores image dateset(folders categorized by class)
├── models/                  # Saves trained models
├── train.py                 # Training script
├── convert_to_onnx.py       # ONNX conversion script
├── requirements.txt         # Dependency package list
└── README.md                # Project documentation
```

Clone the repository on your own computer and prepare your dataset.

```bash
git clone https://github.com/Yushang-Lu/Team2_model.git
cd Team2_model
```

## 2. 🛠️ Environment Setup

### 2.1 Install miniconda

Install Miniconda at [anaconda.com](https://www.anaconda.com)

Create a Miniconda environment and activate it.  

```bash
conda create -n tf python=3.8 -y
conda activate tf
```

### 2.2 Install Dependencies

Install all dependencies:  

```bash
pip install -r requirements.txt
```

*If you prefer to install manually:*  

```bash
pip install tensorflow
pip install tf2onnx onnx onnxruntime
pip install Pillow
# Optional for Apple Silicon:
# pip install tensorflow-metal
```

## 3. 📊 Lightweight 10‑Layer CNN Model  

The model has exactly 10 layers (counting only Conv2D, MaxPooling, Flatten, and Dense – activation layers are not counted separately).  

| Layer | Type          | Details                         |
|-------|---------------|---------------------------------|
| 1     | Conv2D        | 32 filters, 3×3, ReLU           |
| 2     | Conv2D        | 32 filters, 3×3, ReLU           |
| 3     | MaxPooling2D  | 2×2                             |
| 4     | Conv2D        | 64 filters, 3×3, ReLU           |
| 5     | Conv2D        | 64 filters, 3×3, ReLU           |
| 6     | MaxPooling2D  | 2×2                             |
| 7     | Flatten       | -                               |
| 8     | Dense         | 128 units, ReLU                 |
| 9     | Dense         | 64 units, ReLU                  |
| 10    | Dense         | 15 units, Softmax               |

*See model definition:* [train.py](train.py)

**Total parameters:** ~330,000 – very lightweight and fast to train.

## 4. 📈 Training Script (`train.py`)  

[This script](train.py) loads images from the `data/` folder (each subfolder is a class), applies data augmentation, trains the model, and saves the best checkpoint.

**Important:**  

- Place your images in `data/class_name/` – e.g., `data/cat/`, `data/dog/`, ... (15 folders total).  
- The script automatically extracts class labels from folder names (alphabetical order).  

Run this script by:

```bash
python train.py
```

## 5. 📱 Convert Keras Model to ONNX (`convert_to_onnx.py`)  

[This script](convert_to_onnx.py) uses `tf2onnx` to convert the saved `.h5` model to ONNX format.  

Run this script after training to generate `models/model.onnx`.

```bash
python convert_to_onnx.py
```

## 6. 🧪 Test ONNX Inference (`test_onnx.py`)  

[This script](test_onnx.py) loads the ONNX model, preprocesses a single image, runs inference, and prints the predicted class.

**Usage example:**  

```bash
python test_onnx.py models/model.onnx test_images/cat.jpg
```

## 7. 📖 Important Notes  

- **Apple Silicon Users:** Install `tensorflow-metal` to leverage the GPU for faster training.
- **ONNX opset version:** We use `opset=13`, which is widely supported.  
- **Class order:** The order of classes is determined by the alphabetical order of folder names in `data/`. The same order is used in the ONNX inference script when reading class names.  

If you need to adapt the model architecture or data pipeline, simply modify the corresponding scripts. Enjoy building your classifier!

---

## 🤝 Contributing

Contributions are welcome!

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## 🙏 Acknowledgments

- Thanks to the TensorFlow team for the excellent framework
- Thanks to all contributors and users

## 📞 Contact

For questions or suggestions: create an [Issue](https://github.com/Yushang-Lu/Team2_model/issues)

---

⭐️ If this project helps you, please give it a Star!
