
import torch
import argparse
from model import GAT           # TODO check this
# from model_message_passing import GNN
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", help="name of the dataset", type=str)
parser.add_argument("--k", help="number of GNN layers",type=int)
args = parser.parse_args()

dataset = PygGraphPropPredDataset(name = args.dataset)  # ogbg-ppa

split_idx = dataset.get_idx_split() 
train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False)
test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False)

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



model = GAT(hidden_channels=32,num_features=dataset.num_features,num_layers=args.k,num_classes=dataset.num_classes).to(device)   # Note change num_features when needed

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5,weight_decay=5e-1)
criterion = torch.nn.CrossEntropyLoss()

def getMacroF1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='macro')


f1_train_list = []
f1_val_list = []
loss_train_list = []
loss_val_list = []

def train(epoch):
    model.train()

    train_loss = 0

    for data in train_loader:  # Iterate in batches over the training dataset.
        data = data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        # f1_train = getMacroF1(data.y.cpu().numpy(), out.argmax(dim=1).cpu().numpy())
        # f1_train_list.append(f1_train)
        loss = criterion(out, data.y)  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        train_loss += loss.item() * data.num_graphs
    
    train_loss /= len(train_loader.dataset)
    
    if epoch%20 == 0:           # check this?
        f1_train = getMacroF1(data.y.cpu().numpy(), out.argmax(dim=1).cpu().numpy())
        f1_train_list.append(f1_train)
        loss_train_list.append(loss.item())

best_val_f1 = -1

def val(epoch):
    model.eval()
    val_loss_tot = 0
    for data in valid_loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        val_loss = criterion(out, data.y)
        val_loss_tot += val_loss.item() * data.num_graphs

        # check plotting?
        # f1_val = getMacroF1(data.y.cpu().numpy(), out.argmax(dim=1).cpu().numpy())
        # global best_val_f1
        # if f1_val > best_val_f1:
        #     best_val_f1 = f1_val
        # if epoch%20 == 0:
        #     f1_val_list.append(f1_val)
        #     loss_val_list.append(val_loss.item())
    val_loss_tot /= len(valid_loader.dataset)

def test():
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)

            # check this?
            f1_test = getMacroF1(data.y.cpu().numpy(), out.argmax(dim=1).cpu().numpy())
            print('Test F1: ', f1_test)
    
    return f1_test


def plot():
    # plot f1 score for train and val every 20 epochs
    plt.clf()
    plt.plot([20*i for i in range(len(f1_train_list))],f1_train_list, label='train')    # TODO maybe just set ticks at 20
    plt.plot([20*i for i in range(len(f1_val_list))],f1_val_list, label='val')
    plt.legend()
    plt.title("Macro F1 vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.draw()
    plt.savefig(f"Plots/{args.dataset}-{args.k}-perf.png")
    plt.close()

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
    f.write(f"Best Val F1: {best_val_f1:.4f}")



