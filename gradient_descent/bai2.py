def grad_2(x):
    return x**2 - 1

def cost_2(x):
    return (1/3) * x**3 - x

def myGD_bai2(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad_2(x[-1])
        if abs(grad_2(x_new)) < 1e-3: # just a small number
            break
        x.append(x_new)
    return (x, it)