def grad_1(x):
    return 2 * x

def cost_1(x):
    return x**2 - 2

def myGD_bai1(x0, eta):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta * grad_1(x[-1])
        if abs(grad_1(x_new)) < 1e-3: # just a small number
            break
        x.append(x_new)
    return (x, it)