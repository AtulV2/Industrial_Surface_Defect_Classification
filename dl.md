Act as an expert Deep Learning Engineer. I am building an Industrial Surface Defect Classification system. 

Generate the complete, ready-to-run Python codebase for this project across three files: `train.py`, `gradcam.py`, and `app.py`, plus a `requirements.txt`. Do not use placeholder code; write the full, executable implementations.

Here are the strict technical specifications and constraints:

# Environment & Stack
- Run locally. (we are using local pc raher than cloud or google colab) so write code accordingly
- Frameworks: TensorFlow/Keras (for modeling) and Streamlit (for the UI).
- Dataset: NEU Surface Defect Database (1,800 grayscale images, 6 classes: crazing, inclusion, patches, pitted surface, rolled-in scale, scratches).
- Image Resolution: Keep the native 200x200 pixels. Since images are grayscale but MobileNetV2 expects RGB, ensure the data pipeline duplicates the single channel to 3 channels.

# Phase 1: Data Pipeline (in train.py)
- Use `tf.keras.utils.image_dataset_from_directory` with an 80/20 train/validation split.
- Use Keras preprocessing layers for Data Augmentation: Random Rotation, Random Translation, and Random Brightness.

# Phase 2: Hybrid Architecture (in train.py)
Build a single model combining a pre-trained base with custom layers:
- Base: Load `MobileNetV2` (weights='imagenet', include_top=False, input_shape=(200, 200, 3)). Freeze this base model.
- Custom Top Layers: On top of the base model's output, add exactly these layers in this order:
  1. Conv2D (128 filters, 3x3 kernel, padding='same', strides=(1,1), name='custom_conv')
  2. BatchNormalization()
  3. Activation('relu')
  4. MaxPooling2D(pool_size=(2, 2))
  5. GlobalAveragePooling2D()
  6. Dropout(0.5)
  7. Dense(6, activation='softmax')
- Compile with Adam and sparse categorical crossentropy (or categorical depending on how labels are loaded). 
- Train the model and save it as `hybrid_defect_model.keras`.

# Phase 3: Explainability (in gradcam.py)
- Write a reusable Grad-CAM function using `tf.GradientTape`.
- It must extract gradients specifically from the `custom_conv` layer to generate the heatmap.
- Return the superimposed heatmap over the original image.

# Phase 4: Streamlit UI (in app.py)
- Build a web interface using Streamlit.
- Include an image upload widget.
- Upon upload, display the predicted defect class and confidence score clearly at the top.
- Display a side-by-side visual: The original uploaded grayscale image on the left, and the color-mapped Grad-CAM heatmap overlay on the right.

Please provide the exact code for `train.py`, `gradcam.py`, `app.py`, and `requirements.txt`.