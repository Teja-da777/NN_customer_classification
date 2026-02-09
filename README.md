# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Write your own steps

### STEP 2:

### STEP 3:


## PROGRAM

### Name: 
### Register Number:

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
