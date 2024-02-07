import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from skimage.measure import label, regionprops  # Import necessary functions
from PIL import Image
import pathlib
import os
from PIL import Image  

# Set volume parameters  
img_width = 1000
img_height = 1000
num_slices = 400  # Depth 
num_volumes = 1

# Open file
filepath = pathlib.Path("Berea_2d25um_grayscale_filtered.raw")
with filepath.open("rb") as f:

    # Iterate through volumes
    for v in range(num_volumes):

        volume = np.zeros([num_slices, img_height, img_width], dtype=np.uint8)  

        # Read each slice    
        for i in range(num_slices):

            # Read grayscale slice as numpy array
            bytes = f.read(img_width*img_height)        
            slice = np.frombuffer(bytes, dtype=np.uint8).reshape((img_height, img_width))
            
            # Assign to 3D volume
            volume[i] = slice       
        
        # Save slices as images
        for i, slice in enumerate(volume):
            
            im = Image.fromarray(slice)
            im.save(f"slice{i}.jpg")

print("Done extracting volumes and slices!")

# Get code file path
code_dir = os.path.dirname(os.path.abspath(__file__))  

# Set image paths
original_paths = [os.path.join(code_dir, f'slice{i}.jpg') for i in range(400)]  
cropped_paths = [os.path.join(code_dir, f'i{i}.jpg') for i in range(400)]

# Crop dimensions
crop_size = 400

for orig_path, crop_path in zip(original_paths, cropped_paths):

    img = Image.open(orig_path)
    
    # Get width, height
    img_w, img_h = img.size  
    
    # Calculate center coordinate
    center_w, center_h = img_w//2, img_h//2 
    
    # Crop box coords
    left = center_w - crop_size//2     
    right = center_w + crop_size//2
    top = center_h - crop_size//2
    bottom = center_h + crop_size//2
    
    # Crop and save 
    crop_img = img.crop((left, top, right, bottom))
    crop_img.save(crop_path)
    
print('Cropped images saved!')

# Load images from the same location as the code
image_paths = [f"i{i}.jpg" for i in range(400)]  # Assuming images are .jpg format
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

# Corrected function to detect black areas
def calculate_black_ratio(image):

    # Otsu's thresholding
    thresh, bin_img = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Count black pixels
    black_pixels = np.sum(bin_img == 0)

    # Calculate areas 
    black_area = black_pixels
    total_area = image.shape[0]*image.shape[1]

    # Calculate ratio 
    ratio = black_area / total_area

    return ratio

black_ratios = []

for image in images:

    ratio = calculate_black_ratio(image)
    
    black_ratios.append(ratio)

A_b_values = np.array(black_ratios)  
A_r_values = 1 - A_b_values


# Split data into training, validation, and testing sets
X = np.stack([A_b_values, A_r_values], axis=1)
y = np.array([A_b / (A_b + A_r) for A_b, A_r in zip(A_b_values, A_r_values)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# Create and train the machine learning model
model = Sequential([
    Dense(10, activation='relu'),  # First hidden layer with 10 neurons and ReLU activation
    Dense(20, activation='relu'),  # Second hidden layer with 20 neurons and ReLU activation
    Dense(1)                      # Output layer with 1 neuron
])

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=400, validation_data=(X_val, y_val))

# Calculate n_sub for each image using the model
n_sub_values = model.predict(X_test)

# Calculate R2 score
r2 = r2_score(y_test, n_sub_values)
print("R-squared:", r2)

# Calculate average n_sub (n_total)
n_total = np.mean(n_sub_values)
print("Average n_sub (n_total):", n_total)

# Load and prepare test image
i_test = cv2.imread("i_test.jpg", cv2.IMREAD_GRAYSCALE)
ratio = calculate_black_ratio(i_test)

A_b = np.array([ratio]) 
A_r = 1- A_b
X_test = np.stack([A_b, A_r], axis=1)

# Predict n_sub for test image
n_sub = model.predict(X_test)[0][0]
print(f"Predicted n_sub (n) for i_test: {n_sub:0.3f}")
