
import torch
from data_utils import load_data
import argparse
from model import GAT           # TODO check this
# from model_message_passing import GNN
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="name of the dataset", type=str)
parser.add_argument("--k", help="number of GNN layers",type=int)
args = parser.parse_args()
dataset =load_data(args.dataset)
print()
print(f'Dataset: {dataset}:')
print(' Number of GNN layers ', args.k)
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')


gpu=1
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
print(device)

# device='cpu'

data = dataset[0].to(device)  # Get the first graph object.
print()
print(data)
print('===========================================================================================================')

# Stats about the graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')

model_folder = './models/'+args.dataset 

os.makedirs(model_folder, exist_ok=True)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

model = GAT(hidden_channels=32,num_features=dataset.num_features,num_layers=args.k,num_classes=dataset.num_classes).to(device)   # Note change num_features when needed

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,weight_decay=5e-1)
criterion = torch.nn.MSELoss()

def getMacroF1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='macro')


loss_train_list = []
loss_val_list = []

def train():
    model.train()

    for batch_idx, (data1, data2, target) in enumerate(train_loader):
        data1, data2, target = data1.to(device), data2.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        loss_train_list.append(loss.item())

def val():
    model.eval()

    val_loss = 0
    with torch.no_grad():
        for data1, data2, target in val_loader:
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            output = model(data1, data2)
            val_loss += criterion(output, target).item()  # sum up batch loss

def test():
    model.eval()

    test_loss = 0
    with torch.no_grad():
        for data1, data2, target in test_loader:
            data1, data2, target = data1.to(device), data2.to(device), target.to(device)
            output = model(data1, data2)
            test_loss += criterion(output, target).item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)

def plot():

    # plot loss for train and val every 20 epochs
    plt.clf()
    plt.plot([20*i for i in range(len(loss_train_list))],loss_train_list, label='train')
    plt.plot([20*i for i in range(len(loss_val_list))],loss_val_list, label='val')
    plt.legend()
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.draw()
    plt.savefig(f"Plots/{args.dataset}-{args.k}-loss.png")
    plt.close()

num_epochs = 100
best_val_loss = -1
bestModel = None
for epoch in range(num_epochs):
    loss = train(epoch)
    val_loss, val_acc, f1_val = val(epoch)
    if best_val_loss == -1 or val_loss < best_val_loss:
        best_val_loss = val_loss
        bestModel = model
        torch.save(model.state_dict(), model_folder+'/bestval.pth')
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {f1_val:.4f}')

best_test_f1 = -1
with torch.no_grad():
    model.load_state_dict(torch.load(model_folder+'/bestval.pth'))
    best_test_f1, test_acc = test()
    print(f'Test Accuracy: {test_acc:.4f}')

plot()

# write in a file
with open(f'Results/{args.dataset}-{args.k}.txt', 'w') as f:
    # best test f1
    f.write(f"Best Test F1(at best val): {best_test_f1:.4f}\n")
    # best val f1
    f.write(f"Best Val MSE: {best_val_mse:.4f}")



