# PCOS-Detection-using-Ultrasound-Images-with-ResNet-50
This project detects PCOS (Polycystic Ovary Syndrome) from ultrasound images using deep learning. I used ResNet-50 as the main model, compared it with AlexNet and SVM, and performed EDA using Python libraries like OpenCV and Matplotlib. The model was trained, evaluated, and predictions were made using the Gemini API.

Project Details
Dataset: Ultrasound images divided into 'infected' and 'not_infected' folders under 'train' and 'test' directories.

Exploratory Data Analysis (EDA):

Counted number of images.

Analyzed image size distribution.

Visualized pixel intensity and color distributions.

Models Used:

ResNet-50 (Main model with best accuracy)

AlexNet (For comparison)

Support Vector Machine (SVM) (For comparison)

Tools and Libraries:

Python, TensorFlow, OpenCV, Matplotlib, Seaborn, NumPy

API Integration:

Used Gemini API to train the model, save it, and make predictions for new images.

How to Run
Preprocess the images.

Train the model using the training set.

Save the trained model.

Evaluate the model on the test set and print the accuracy.

Use the saved model to predict PCOS for new ultrasound images.

Results
ResNet-50 achieved the highest accuracy among all models.
