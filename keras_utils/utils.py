import matplotlib.pyplot as plt


def plot_loss(history):
    loss = history.history.get('loss')
    plt.plot(loss, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
