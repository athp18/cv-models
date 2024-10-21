import torch
import torch.optim as optim
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Function
import conv2d  # compiled CUDA extension

class Conv2D(Function):
    @staticmethod
    def forward(ctx, input, kernel, stride=1, padding=0):
        #we need to make sure were on the right device
        input = input.contiguous()
        kernel = kernel.contiguous()
        
        batch_size, input_channels, input_height, input_width = input.shape
        output_channels, _, kernel_size, _ = kernel.shape

        # compute output shape
        output_height = (input_height - kernel_size + 2 * padding) // stride + 1
        output_width = (input_width - kernel_size + 2 * padding) // stride + 1

        # allocate space for the output
        output = torch.zeros((batch_size, output_channels, output_height, output_width), device=input.device)

        # call the custom CUDA kernel
        conv2d.conv2d_kernel(input, output, kernel, 
                                    input_channels, output_channels, 
                                    input_height, input_width, 
                                    kernel_size, stride, padding)

        return output


#conv 2d layer
class Conv2D_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv2D_Layer, self).__init__()
        self.kernel = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return Conv2D.apply(x, self.kernel, self.stride, self.padding)


# lenet model
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = Conv2D_Layer(1, 6, 5)  # Use custom Conv2D layer
        self.conv2 = Conv2D_Layer(6, 16, 5)  # Use custom Conv2D layer
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))     
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))     
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 4 * 4)    
        x = F.relu(self.fc1(x))       
        x = F.relu(self.fc2(x))       
        x = self.fc3(x)
        return x


# Training loop
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Zero the gradients
        output = model(data)   # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max logprob
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = LeNet().to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        train(model, device, train_loader, optimizer, loss, epoch)
        test(model, device, test_loader, loss)


if __name__ == '__main__':
    main()
