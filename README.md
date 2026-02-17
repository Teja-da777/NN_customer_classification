# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="869" height="713" alt="Screenshot 2026-02-06 184805" src="https://github.com/user-attachments/assets/6e808709-6ddc-4642-906f-b2f4c05dbc23" />

## DESIGN STEPS:

### STEP 1:
Understand the classification task and identify input and output variables.

### STEP 2:
Gather data, clean it, handle missing values, and split it into training and test sets.
### STEP 3:
Normalize/standardize features, encode categorical labels, and reshape data if needed.
### STEP 4:
Choose the number of layers, neurons, and activation functions for your neural network.

### STEP 5:
Select a loss function (e.g., binary cross-entropy), optimizer (e.g., Adam), and metrics (e.g., accuracy).


### STEP 6:
Feed training data into the model, run multiple epochs, and monitor the loss and accuracy.

### STEP 7:
Save the trained model, export it if needed, and deploy it for real-world use.


## PROGRAM

### Name: THEJESWARAN M
### Register Number: 212223240168

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1=nn.Linear(input_size,16)
        self.fc2=nn.Linear(16,8)
        self.fc3=nn.Linear(8,4)

    def forward(self, x):
      x=F.relu(self.fc1(x))
      x=F.relu(self.fc2(x))
      x=self.fc3(x)
      return x
        
# Initialize the Model, Loss Function, and Optimizer
model =PeopleClassifier(input_size=X_train.shape[1])
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.01)
def train_model(model, train_loader, criterion, optimizer, epochs):
   for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      outputs = model(X_batch)
      loss = criterion(outputs, y_batch)
      loss.backward ()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```



## Dataset Information

<img width="990" height="586" alt="Screenshot (56)" src="https://github.com/user-attachments/assets/b2337e12-bda7-4694-b862-cd3468f08500" />

## OUTPUT

### Confusion Matrix
<img width="640" height="480" alt="fig1" src="https://github.com/user-attachments/assets/0f425fd3-0dab-4d5d-baf9-fdc59535f6c9" />

### Classification Report
<img width="676" height="421" alt="classification" src="https://github.com/user-attachments/assets/b399e3de-3306-47e8-8e13-f7811cf2b0a6" />

### New Sample Data Prediction
<img width="676" height="97" alt="Screenshot (54)" src="https://github.com/user-attachments/assets/79d8bc3e-615d-4aa7-8716-41b6c48c0320" />

## RESULT
Thus, a neural network classification model for the given dataset as been created successfully.
