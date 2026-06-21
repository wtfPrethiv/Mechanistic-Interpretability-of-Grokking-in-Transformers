import torch

def generate_dataset():

    P = 113
    x = []
    y = []

    for a in range(P):
        for b in range(P):
            x.append([a, b, P])
            y.append((a + b) % P)

    return {
        "x": torch.tensor(x, dtype=torch.long),
        "y": torch.tensor(y, dtype=torch.long)
    }

data = generate_dataset()

torch.save(data, "data/dataset.pt")
