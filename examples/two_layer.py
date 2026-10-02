import vectorcore as vc

x = vc.Tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
y = vc.Tensor([[0.0], [0.0], [0.0], [1.0], [1.0]])
first, second = vc.Linear(1, 4, 1), vc.Linear(4, 1, 2)
parameters = [first.weight, first.bias, second.weight, second.bias]

for epoch in range(401):
    prediction = second(first(x).relu()).sigmoid()
    loss = vc.mse_loss(prediction, y)
    if epoch % 100 == 0:
        print(f"epoch={epoch} loss={loss.item():.6f}")
    loss.backward()
    vc.sgd_step(parameters, 0.25)
    vc.zero_grad(parameters)
