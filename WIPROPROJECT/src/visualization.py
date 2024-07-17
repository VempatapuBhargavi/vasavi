import matplotlib.pyplot as plt

def plot_training_validation_metrics(train_losses, valid_losses, train_accuracies, valid_accuracies, num_epochs):
    plt.figure(figsize=(12, 5))

    # Plotting Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, 'bo-', label='Training loss')
    plt.plot(range(1, num_epochs + 1), valid_losses, 'r--', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, 'bo-', label='Training accuracy')
    plt.plot(range(1, num_epochs + 1), valid_accuracies, 'r--', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()