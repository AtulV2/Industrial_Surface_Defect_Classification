# Phase 2 Plan: Transfer Learning Model Architecture and Training

**Phase:** 2 — Model Training  
**Script:** `src/train.py`  
**Status:** In Progress (Autopilot Execution)  
**Date:** 2026-11-04  

---

## 1. Objectives & Scope
- Build and train a hybrid CNN model for multi-class surface defect classification (6 classes).
- Training dataset: 7,200 images from `data/augmented/train/` (balanced across classes).
- Validation dataset: 360 images from `data/raw/validation/images/`.
- Architecture: Pretrained `MobileNetV2` (frozen base) + custom Conv2D/BatchNorm/ReLU/MaxPool/GlobalAvgPool/Dropout top.
- Save best model weights as `models/hybrid_defect_model.keras`.
- Generate training metrics visualization in `outputs/training_history.png`.

---

## 2. Dataset Pipeline Specification
- Utilizes `tf.keras.utils.image_dataset_from_directory`:
  - `image_size=(200, 200)`
  - `batch_size=32`
  - `color_mode='rgb'` (handles grayscale/RGB format uniformly)
  - `label_mode='int'` (for `sparse_categorical_crossentropy`)
  - `shuffle=True` for train, `shuffle=False` for validation
  - `seed=42`
- Preprocessing:
  - Normalization via `tf.keras.layers.Rescaling(1./255)`.
  - Pipelining with `prefetch(buffer_size=tf.data.AUTOTUNE)`.

---

## 3. Exact Architecture Specification
- **Base Model**:
  - `MobileNetV2(weights='imagenet', include_top=False, input_shape=(200, 200, 3))`
  - `base_model.trainable = False` (all base layers frozen).
- **Custom Top Head (Functional API)**:
  - `x = Conv2D(128, (3, 3), padding='same', strides=(1, 1), name='custom_conv')(base_model.output)`
  - `x = BatchNormalization()(x)`
  - `x = Activation('relu')(x)`
  - `x = MaxPooling2D(pool_size=(2, 2))(x)`
  - `x = GlobalAveragePooling2D()(x)`
  - `x = Dropout(0.5)(x)`
  - `outputs = Dense(6, activation='softmax')(x)`
- **Model**:
  - `model = Model(inputs=base_model.input, outputs=outputs)`

---

## 4. Compilation, Callbacks & Training
- Optimizer: `Adam(learning_rate=1e-3)`
- Loss: `sparse_categorical_crossentropy`
- Metrics: `['accuracy']`
- Callbacks:
  - `ModelCheckpoint('models/hybrid_defect_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')`
  - `EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy', mode='max')`
- Epochs: 30 (with early termination if validation accuracy plateaus).

---

## 5. Verification & Deliverables
- Model summary printed to console.
- Final train & validation accuracy evaluated and printed.
- Training loss and accuracy plots saved to `outputs/training_history.png`.
- Model artifact saved to `models/hybrid_defect_model.keras`.
- Governance logs:
  - `.ai/04_audits/AUDIT_003.md`
  - `.ai/03_checkpoints/CHECKPOINT_003.md`
  - `.ai/05_chat_logs/LOG_003_phase_2_decisions.md`
  - Updated `.ai/CURRENT_STATE.md`
- Git commit: `feat(phase-2): implement transfer learning defect classifier with mobilenetv2 hybrid`

