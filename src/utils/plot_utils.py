import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_embeddings(embeddings, labels, title):
    plt.figure(figsize=(10, 10))
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        plt.scatter(x, y, c='b')
        plt.text(x, y, label, fontsize=9)
    plt.title(title)
    plt.grid(True)
    plt.show()