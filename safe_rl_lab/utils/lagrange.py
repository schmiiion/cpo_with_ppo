import torch

class Lagrange:
    def __init__(self, cost_limit, lagrangian_multiplier_init, lambda_lr, lambda_optimizer='adam'):
        self.cost_limit = cost_limit
        self.lambda_lr = lambda_lr

        # The multiplier is a learnable parameter
        self.lagrangian_multiplier = torch.nn.Parameter(
            torch.tensor(lagrangian_multiplier_init, dtype=torch.float32),
            requires_grad=True
        )

        # Optimizer for lambda (Dual variable descent)
        if lambda_optimizer == 'adam':
            self.lambda_optimizer = torch.optim.Adam([self.lagrangian_multiplier], lr=lambda_lr)
        else:
            self.lambda_optimizer = torch.optim.SGD([self.lagrangian_multiplier], lr=lambda_lr)

    def update_lagrange_multiplier(self, Jc):
        """
        Update lambda to minimize the Lagrangian: L = ... + lambda * (Jc - limit)
        Since we want to Maximize L w.r.t Lambda (Dual Ascent), or strictly enforce constraint:
        loss = - lambda * (Jc - limit)  <-- Minimize this
        """
        # Ensure Jc is a tensor
        Jc = torch.as_tensor(Jc, dtype=torch.float32)

        # We want lambda to grow if Jc > limit
        # Loss for optimizer: lambda * (limit - Jc)
        # Gradient descent on this will increase lambda if Jc > limit
        loss = self.lagrangian_multiplier * (self.cost_limit - Jc)

        self.lambda_optimizer.zero_grad()
        loss.backward()
        self.lambda_optimizer.step()

        # Enforce lambda >= 0 (Projection)
        self.lagrangian_multiplier.data.clamp_(min=0.0)

        return self.lagrangian_multiplier.item()