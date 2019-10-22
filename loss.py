

def focal_loss(gamma, x):
    return -(1 - x) ** gamma * x.log()