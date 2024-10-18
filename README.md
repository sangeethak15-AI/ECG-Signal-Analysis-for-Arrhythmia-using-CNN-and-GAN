# ECG-Signal-Analysis-for-Arrhythmia-using-CNN-and-GAN
## Hardware Requirements:
1.PC Processor - 11th Gen Intel
2.System type :64-bit processor
## SoftwarenRequirements:
1. Jupyter Notebook
2. Python Libraries
## Implementation:
1. The project is implemented by python using the following libraries:
A. Jupyter Notebook
B. Tensorflow
2. The project includes Convolution Neural Network that can be used to extract the patterns.
3. This model first applies 1D convolution layers to extract features from ECG signals.
4. Max pooling is used to to downsample the data. T
5. Two LSTM layers :  capture temporal dependencies in the sequential data, with dropout added to prevent overfitting. The output of the LSTM is fed into dense layers for final feature transformation. 
6. A sigmoid-activated dense layer provides the binary classification output (arrhythmia detection).
   
## System Architecture:
![Arch diagram](https://github.com/user-attachments/assets/5fb29ccc-fbff-431a-8572-e72442b7359e)


## Code
### Model of GAN:
```
# Set random seed for reproducibility
np.random.seed(60)
tf.random.set_seed(60)

# Create a simple dataset (replace this with your ECG dataset)
# Assuming each sample is a 1D array of shape (n_samples, n_features)
# For demonstration, let's generate random data
n_samples = 1000  # Original dataset size
n_features = 10   # Number of features (like ECG values)

# Example ECG dataset
real_data = np.random.rand(n_samples, n_features)

# Normalize the dataset
#real_data = (real_data - 0.5) * 2  # Scale to [-1, 1]

# Define the GAN components
# Generator
def build_generator(latent_dim, n_features):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_dim=latent_dim),
        layers.Dense(256, activation='relu'),
        layers.Dense(n_features, activation='tanh'),  # Output should be between -1 and 1
    ])
    return model

# Discriminator
def build_discriminator(n_features):
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(n_features,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='relu'), 
    ])
    return model

# Compile GAN
latent_dim = 100  # Size of the random noise vector
generator = build_generator(latent_dim, n_features)
discriminator = build_discriminator(n_features)

# Define the optimizer
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create the GAN model
discriminator.trainable = False  # Freeze the discriminator during GAN training
gan_input = layers.Input(shape=(latent_dim,))
generated_data = generator(gan_input)
gan_output = discriminator(generated_data)
gan = keras.Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')
```
###  Model of CNN-LSTM:
```
def build_cnn_lstm_model(input_shape):
    model = keras.Sequential([
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification for arrhythmia
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
```
## Output:
![image](https://github.com/user-attachments/assets/2665f194-f9a2-482a-a24e-fc5216b8b9d4)
![WhatsApp Image 2024-10-18 at 10 34 52_1305260e](https://github.com/user-attachments/assets/8f07bde5-0daa-46bf-899a-84c42da5d2a9)
