
import torch
from data_utils import load_data
import argparse
from model import GAT           # TODO check this
# from model_message_passing import GNN
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
from torch_geometric.data import DataLoader


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
print(dataset)
if args.dataset == 'AIDS':
    # train first 420 graphs and validate on the rest
    train_indices = range(420)
    valid_indices = range(420, len(dataset))
    split_idx = {'train': train_indices, 'valid': valid_indices}
    dataset.split_idx = split_idx
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx['valid']], batch_size=32)        # what to do here?
elif args.dataset == 'LINUX':
    # train first 800 graphs and validate on the rest
    train_indices = range(800)
    valid_indices = range(800, len(dataset))
    split_idx = {'train': train_indices, 'valid': valid_indices}
    dataset.split_idx = split_idx
    train_loader = DataLoader(dataset[split_idx['train']], batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset[split_idx['valid']], batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset[split_idx['valid']], batch_size=32)        # what to do here?
else:
    print('Dataset not found')
    exit(0)


model = GAT(hidden_channels=32,num_features=dataset.num_features,num_layers=args.k,num_classes=dataset.num_classes).to(device)   # Note change num_features when needed

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,weight_decay=5e-1)
criterion = torch.nn.MSELoss()


loss_train_list = []
loss_val_list = []

def train(epoch):
    model.train()

    # use all pair of graphs in train_loader

    train_loss = 0

    for data1 in train_loader:
        for data2 in train_loader:
            data1, data2 = data1.to(device), data2.to(device)
            optimizer.zero_grad()
            output = model(data1.x, data1.edge_index, data1.batch,  data2.x, data2.edge_index, data2.batch)
            target = dataset.ged(data1.i, data2.i)  # check this? Does it work with batching?
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
    train_loss /= len(train_loader.dataset*len(train_loader.dataset))
    # print epoch and loss
    print('Train Epoch {} Loss: {:.6f}'.format(epoch, train_loss))
    loss_train_list.append(train_loss)

def val(epoch):
    model.eval()

    val_loss = 0
    with torch.no_grad():
        for data1 in valid_loader:
            for data2 in valid_loader:
                data1, data2 = data1.to(device), data2.to(device)
                output = model(data1.x, data1.edge_index, data1.batch,  data2.x, data2.edge_index, data2.batch)
                target = dataset.ged(data1.i, data2.i)
                val_loss += criterion(output, target).item()  # sum up batch loss
        
    val_loss /= len(valid_loader.dataset*len(valid_loader.dataset))
    print('Val Epoch {} Loss: {:.6f}'.format(epoch, val_loss))
    loss_val_list.append(val_loss)

def test():
    model.eval()

    test_loss = 0
    with torch.no_grad():
        for data1 in test_loader:
            for data2 in test_loader:
                data1, data2 = data1.to(device), data2.to(device)
                output = model(data1.x, data1.edge_index, data1.batch,  data2.x, data2.edge_index, data2.batch)
                target = dataset.ged(data1.i, data2.i)
                test_loss += criterion(output, target).item()  # sum up batch loss
    
    test_loss /= len(test_loader.dataset*len(test_loader.dataset))
    print('Test Loss: {:.6f}'.format(test_loss))
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



