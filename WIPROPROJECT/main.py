import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from model.simple_cnn import SimpleCNN
from src.evaluation import evaluate_model
from src.visualization import plot_training_validation_metrics
from src.train import train_model, validate_model
from sklearn.metrics import precision_score, recall_score, f1_score

def main():
    # Define your dataset directory
    data_dir = 'C:/Users/BHARGAVI/PycharmProjects/pythonProject/WIPROPROJECT/Dataset'

    # Check the contents of the dataset directory
    print(os.listdir(data_dir))

    # Define classes for train and validation sets
    classes_train = os.listdir(os.path.join(data_dir, "train"))
    classes_valid = os.listdir(os.path.join(data_dir, "validation"))

    print(f'Train Classes - {classes_train}')
    print(f'Validation Classes - {classes_valid}')

    # Define transforms for training and validation datasets
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # Create datasets using ImageFolder
    train_ds = ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
    valid_ds = ImageFolder(os.path.join(data_dir, 'validation'), transform=valid_transforms)

    # Define batch size and create data loaders
    batch_size = 32
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size * 2, num_workers=0, pin_memory=True)

    # Check if CUDA (GPU support) is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model = SimpleCNN(num_classes=len(train_ds.classes)).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Define number of epochs for training
    num_epochs = 50

    # Lists to store metrics for plotting
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    # Train the model
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train_model(model, train_dl, criterion, optimizer, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)


        # Optionally, validate the model after each epoch
        valid_loss, valid_accuracy = validate_model(model, valid_dl, criterion, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.4f}')
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

    # Save the trained model
    save_path = ('C:/Users/BHARGAVI/PycharmProjects/pythonProject/WIPROPROJECT/Dataset')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully at '{save_path}'.")

    # Evaluate the model on the validation set
    validation_accuracy = evaluate_model(model, valid_dl, device)
    print(f'Validation Accuracy: {validation_accuracy:.4f}')


    # Plot the training and validation metrics
    plot_training_validation_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies, num_epochs)
    
    print("Training, evaluation, and plotting completed.")

if __name__ == "__main__":
    main()