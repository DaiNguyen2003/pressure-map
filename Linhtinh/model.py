#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
import numpy as np

# Dữ liệu giả lập
X_train = np.random.uniform(3, 10, size=(500, 1))  # giá trị ô giữa
Y_train = np.exp(-0.5 * X_train) * 5 + np.random.normal(0, 0.2, (500, 8))  # giảm phi tuyến có nhiễu

# MLP model
model = Sequential([
    Dense(64, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(8)  # 8 ô xung quanh
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, epochs=50, batch_size=16)
