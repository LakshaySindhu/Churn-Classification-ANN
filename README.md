# Artificial Neural Network (ANN) for Churn Prediction

This project implements an Artificial Neural Network (ANN) to predict customer churn using the `Churn_Modelling.csv` dataset. The workflow covers data preprocessing, model building, training, evaluation, and logging with TensorBoard.

## Project Structure

```
ANN/
├── app.py                      # Main application script
├── Churn_Modelling.csv         # Dataset
├── experiments.ipynb           # Experimentation notebook
├── label_encoder_gender.pkl    # Saved label encoder for gender
├── model.h5                    # Trained ANN model
├── onehot_encoder_geo.pkl      # Saved one-hot encoder for geography
├── predictions.ipynb           # Prediction notebook
├── requirements.txt            # Python dependencies
├── scaler.pkl                  # Saved scaler for feature normalization
├── TensorFlow_M1_Issue_Fix_Formatted.pdf # TensorFlow M1 Mac issue fix
├── Tensorflow.ipynb            # TensorFlow workflow notebook
└── logs/                       # TensorBoard logs
```

## Steps in ANN Implementation

1. **Sequential Network:** Build the model using a sequential architecture.
2. **Dense Layers:** Add hidden neurons using dense layers.
3. **Activation Functions:** Use activation functions like `sigmoid`, `tanh`, `relu`, or `leakyrelu`.
4. **Optimizer:** Apply backpropagation using optimizers (e.g., Adam, SGD).
5. **Loss Function:** Define the loss function (e.g., binary crossentropy).
6. **Metrics:** Track metrics such as accuracy, MSE, and MAE.
7. **Training & Logging:** Train the model and log results using TensorBoard.

## Getting Started

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Run the main script:**
   ```
   python app.py
   ```

3. **View TensorBoard logs:**
   ```
   tensorboard --logdir=logs/
   ```

## Notebooks

- `experiments.ipynb`: Data exploration and model experimentation.
- `predictions.ipynb`: Making predictions with the trained model.
- `Tensorflow.ipynb`: TensorFlow workflow and troubleshooting.

## Notes

- Encoders and scalers are saved as `.pkl` files for reuse.
- Model is saved as `model.h5` after training.
- TensorBoard logs are stored in the `logs/` directory.

## License

This project is for educational purposes.