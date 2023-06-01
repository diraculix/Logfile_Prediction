import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the architecture of the neural network
class PencilBeamModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PencilBeamModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# path to training data
dpath = r'/home/luke/Scripts/Logfile_Prediction/datasets'
record = pd.read_csv(os.path.join(dpath, 'patient_1676348_records_data.csv')) 
delta = pd.read_csv(os.path.join(dpath, 'patient_1676348_log-plan_delta.csv'))  # should be same shape
delta.drop(columns=['GANTRY_ANGLE'], inplace=True) 
if len(record) == len(delta):
    data = pd.concat([record, delta], axis=1)
else:
    raise ValueError('Datasets must be of same length')

data['X_PLAN(mm)'] = np.round(data['X_POSITION(mm)'] - data['DELTA_X(mm)'], 3)
data['Y_PLAN(mm)'] = np.round(data['Y_POSITION(mm)'] - data['DELTA_Y(mm)'], 3)
data['DIST_TO_ISO(mm)'] = np.sqrt(data['X_PLAN(mm)'] ** 2 + data['Y_PLAN(mm)'] ** 2)
data['MU_PLAN'] = data['MU'] - data['DELTA_MU']
data['ENERGY(MeV)'] = np.round(data['LAYER_ENERGY(MeV)'], 1)
data.dropna(inplace=True)
print(len(data))
data_minmax = data.copy()
for col in data.columns:
    try: data_minmax[col] = (data_minmax[col] - data_minmax[col].min()) / (data_minmax[col].max() - data_minmax[col].min())
    except: pass

# Define the hyperparameters
input_size = 5  # Number of input features: spot x-position, spot y-position, monitor units, spot energy, gantry angle
hidden_size = 18  # Number of neurons in the hidden layer
output_size = 3  # Number of output features: predicted delivery errors in dx, dy, dMU

# Create an instance of the model
model = PencilBeamModel(input_size, hidden_size, output_size)

# Define the loss function
criterion = nn.MSELoss()
  
# view normalized data
# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load the data from pandas DataFrames (replace with your own data loading code)
df_data = data[['X_POSITION(mm)', 'Y_POSITION(mm)', 'MU_PLAN', 'GANTRY_ANGLE', 'ENERGY(MeV)']]
df_labels = data[['DELTA_X(mm)', 'DELTA_Y(mm)', 'DELTA_MU']]

# Convert data and labels into tensors
data = torch.tensor(df_data.values, dtype=torch.float32)
labels = torch.tensor(df_labels.values, dtype=torch.float32)

# Convert data and labels into a TensorDataset
dataset = TensorDataset(data, labels)

# Create data loaders for cross-validation
batch_size = 2600
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the final model on the entire dataset
final_num_epochs = 100
losses = []
for epoch in range(final_num_epochs):
    for batch_data, batch_labels in data_loader:
        # model.train()
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print('Epoch:', epoch + 1, 'Loss:', loss.item())

    if epoch > 20:
        optimizer.param_groups[0]['lr'] = 0.0001
    
    losses.append(loss.item())

plt.plot(range(final_num_epochs), losses, label='MSE Loss')
plt.legend()
plt.show()

# Save the final trained model to a file
torch.save(model.state_dict(), 'final_model.pth')
