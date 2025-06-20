# 🤖 Smart Waste Sorter

An AI-powered application that uses computer vision to classify waste in real-time through a webcam. The system identifies objects as either **Recyclable** or **Non-Recyclable** and provides instant visual feedback.


*(This is a placeholder GIF. You can create your own using a tool like [ScreenToGif](https://www.screentogif.com/) and replace the link!)*

---

## ✨ Features

-   **Real-Time Classification:** Identifies waste items instantly via a live webcam feed.
-   **Deep Learning Model:** Built on a `MobileNetV2` architecture, trained to distinguish between two primary categories.
-   **Intuitive Visual Feedback:** The screen displays the classification result with a confidence score and a color-coded bounding box (🟢 Green for Recyclable, 🔴 Red for Non-Recyclable).
-   **Simple & Modular:** Easy-to-understand code structure, with separate scripts for training and execution.

---

## 🛠️ Technology Stack

-   **Python**
-   **TensorFlow** (for building and training the deep learning model)
-   **OpenCV-Python** (for camera access and image processing)
-   **NumPy** (for numerical operations)
-   **Scipy** (for data augmentation)

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

-   Python 3.9+
-   Git
-   A webcam connected to your computer

### Installation

1.  **Clone the repository and navigate into the directory:**
    ```bash
    git clone https://github.com/Zaid2044/Smart-Waste-Sorter.git
    cd Smart-Waste-Sorter
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    *(On Windows with Git Bash, you might need to use `source venv/Scripts/activate`)*

3.  **Install the required dependencies:**
    ```bash
    pip install tensorflow opencv-python numpy python-dotenv matplotlib scipy
    ```

---

## ⚡ Usage

The pre-trained model (`waste_sorter_model.keras`) is already included.

### 1. Run the Sorter

To run the main application and start the real-time classification with your webcam:

```bash
python3 run_sorter.py