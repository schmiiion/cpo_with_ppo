import torch.nn as nn

class PIDLagrange:
    """
        PID Controller for the Lagrangian Multiplier.
        Formula: lambda = max(0, Kp * error + Ki * integral + Kd * derivative)
    """
    def __init__(self, cost_limit, kp, ki, kd):
        self.cost_limit = cost_limit # in Algorithm from Stooke et al. called "d"
        self.kp = kp
        self.ki = ki
        self.kd = kd

        #state
        self.I = 0
        self.Jc_prev = 0.0
        self.lam = 0.0


    def update(self, Jc):
        """
        Update the penalty based on the current cost.
        Returns the new penalty lambdas
        """
        delta = Jc - self.cost_limit

        #calculate derivative
        derivative = max(0.0, Jc - self.Jc_prev)

        #calculate Integral
        self.I = max(0.0, self.I + delta)
        print(f"integral value: {self.I}")

        # compute lambda_t+1
        self.lam = max(0.0, self.kp * delta + self.ki * self.I + self.kd * derivative)

        self.Jc_prev = Jc

        return self.lam
