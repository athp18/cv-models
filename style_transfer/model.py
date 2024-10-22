import torch
import torch.optim as optim
from torch import nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import copy

# Load and preprocess images
def load_image(image_path, max_size=400, shape=None):
    image = Image.open(image_path)
    
    # Resize image if it's too large
    size = max_size if max(image.size) > max_size else max(image.size)
    
    if shape is not None:
        size = shape
        
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),  # Remove alpha channel if exists
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image

def im_convert(tensor):
    """Convert a tensor to an image."""
    image = tensor.clone().detach()
    image = image.squeeze(0)  # Remove batch dimension
    image = image.numpy().transpose(1, 2, 0)  # Transpose to HWC
    image = image * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # Denormalize
    image = image.clip(0, 1)  # Clip to valid range
    
    return image

# VGG19 Model for extracting features
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg_features = models.vgg19(pretrained=True).features
        
        # Define layers for content and style
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        self.selected_layers = {
            '0': 'conv_1',
            '5': 'conv_2',
            '10': 'conv_3',
            '19': 'conv_4',
            '28': 'conv_5'
        }
        
    def forward(self, x):
        content_features = []
        style_features = []
        
        for name, layer in self.vgg_features._modules.items():
            x = layer(x)
            if name in self.selected_layers:
                if self.selected_layers[name] in self.style_layers:
                    style_features.append(x)
                if self.selected_layers[name] in self.content_layers:
                    content_features.append(x)
                    
        return content_features, style_features

# Gram matrix for style representation
def gram_matrix(tensor):
    _, n_filters, h, w = tensor.size()
    tensor = tensor.view(n_filters, h * w)
    G = torch.mm(tensor, tensor.t())
    return G.div(n_filters * h * w)

# Loss calculation
class StyleContentLoss(nn.Module):
    def __init__(self, content_img, style_img):
        super(StyleContentLoss, self).__init__()
        self.content_img = content_img
        self.style_img = style_img
        self.vgg = VGG().to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Extract features of content and style images
        self.content_features, _ = self.vgg(self.content_img)
        _, self.style_features = self.vgg(self.style_img)
        self.style_grams = [gram_matrix(feature) for feature in self.style_features]

    def forward(self, x):
        content_loss = 0.0
        style_loss = 0.0
        
        # Extract content and style features from generated image
        content_features, style_features = self.vgg(x)
        content_loss = torch.mean((content_features[0] - self.content_features[0]) ** 2)

        # Style loss using Gram matrices
        for gen_feat, style_gram in zip(style_features, self.style_grams):
            G = gram_matrix(gen_feat)
            style_loss += torch.mean((G - style_gram) ** 2)
        
        total_loss = content_loss * content_weight + style_loss * style_weight
        return total_loss

# Hyperparameters
content_weight = 1e4  # Weight for content loss
style_weight = 1e2    # Weight for style loss
num_steps = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load content and style images
content_image = load_image("path_to_content_image.jpg").to(device)
style_image = load_image("path_to_style_image.jpg", shape=content_image.shape[-2:]).to(device)

# Initialize generated image
generated_image = content_image.clone().requires_grad_(True)

# Optimizer
optimizer = optim.Adam([generated_image], lr=0.003)

# Loss model
loss_model = StyleContentLoss(content_image, style_image)

for step in range(num_steps):
    optimizer.zero_grad()
    loss = loss_model(generated_image)
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        print(f"Step {step}, Total loss: {loss.item()}")
        plt.imshow(im_convert(generated_image))
        plt.show()

# Save the final output
final_image = im_convert(generated_image)
plt.imshow(final_image)
plt.show()
