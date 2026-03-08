# Hand Gesture Recognition System

## Overview

This project is a **Hand Gesture Recognition System** that detects and recognizes **up to 15 different hand gestures** in real time using a webcam.

The gestures are identified using **predefined finger position patterns** instead of a trained machine learning model. Each gesture is represented using values that indicate whether a finger is **open (1) or closed (0)**.

The system is built using **Python for gesture detection** and **HTML/CSS for the user interface**.

## Features

* Recognizes **up to 15 hand gestures**
* Supports **both left and right hands**
* Real-time gesture detection using a webcam
* Rule-based gesture recognition using finger patterns
* Simple web interface

## Technologies Used

* Python
* HTML
* CSS

## How Gesture Recognition Works

The program checks the **state of each finger** and represents it as a pattern.

Example:

```
0,0,0,0,0 → Fist
1,1,1,1,1 → Open Palm
0,1,0,0,0 → One Finger
```

These patterns are compared with predefined gesture values in the Python code to identify the gesture.

## Important Note

The system recognizes gestures from **both hands**, but **left hand gestures are detected correctly only when the back side of the hand is shown (opposite orientation)**. This occurs because the finger position pattern changes depending on the hand orientation.

## Project Structure

```
project-folder
│
├── app.py
├── requirements.txt
├── templates
│   └── index.html
├── static
│   └── style.css
└── README.md
```

## Installation

1. Clone the repository
2. Install the required libraries using the `requirements.txt` file

```
pip install -r requirements.txt
```

3. Run the application

```
python app.py
```

4. Open the application and start showing gestures to the webcam.

## Future Improvements

* Improve left-hand gesture detection accuracy
* Add support for more gestures
* Enhance the user interface

## Author

Hand Gesture Recognition project developed using Python and web technologies.
