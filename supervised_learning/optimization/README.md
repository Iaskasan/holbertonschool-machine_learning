❓ What is a hyperparameter?

A hyperparameter is like a setting knob you choose before training a model.
They are not learned from the data; instead, they control how the training happens.

Examples:

Learning rate (size of the update steps in gradient descent)

Number of hidden layers in a neural network

Batch size during training

❓ How and why do you normalize your input data?

Normalize = put all features on a similar scale.

Why?
If one feature (like age = 20–80) and another (like income = 1000–100000) have very different ranges, the model struggles.

How:

Min–max scaling:

𝑥
′
=
𝑥
−
min
⁡
(
𝑥
)
max
⁡
(
𝑥
)
−
min
⁡
(
𝑥
)
x
′
=
max(x)−min(x)
x−min(x)
	​


Standardization (z-score):

𝑥
′
=
𝑥
−
𝜇
𝜎
x
′
=
σ
x−μ
	​


This makes training faster and more stable.

❓ What is a saddle point?

A saddle point is like the seat of a horse saddle:

Going forward/back, it curves down (like a valley).

Going left/right, it curves up (like a hill).

It’s not a true minimum, but gradient descent can get stuck there because the slope looks flat.

❓ What is stochastic gradient descent (SGD)?

Gradient descent normally computes gradients using the whole dataset (slow).

Stochastic Gradient Descent (SGD):

Uses just 1 example at a time.

Much faster but noisy.

❓ What is mini-batch gradient descent?

A compromise between full batch and SGD:

Uses small groups of samples (mini-batches) like 32 or 128 examples.

Faster than full batch, less noisy than pure SGD.

The most common approach in practice.

❓ What is a moving average? How do you implement it?

A moving average smooths out values over time.

Example: Loss values [5, 4, 6, 3] with window = 2
→ Moving averages = [4.5, 5, 4.5]

Python:

def moving_average(values, window):
    ma = []
    for i in range(len(values) - window + 1):
        ma.append(sum(values[i:i+window]) / window)
    return ma

❓ What is gradient descent with momentum?

Think of a ball rolling down a hill. Instead of stopping at every bump, it builds speed.

Rule:

v = beta * v + (1 - beta) * gradient
weights = weights - alpha * v


beta = how much past gradients matter.

Momentum helps escape small bumps and speeds up training.

❓ What is RMSProp? How do you implement it?

RMSProp (Root Mean Square Propagation):

Keeps a moving average of squared gradients.

Scales learning rates to avoid exploding or vanishing updates.

Rule:

s = beta * s + (1 - beta) * (gradient**2)
weights = weights - alpha * gradient / (sqrt(s) + epsilon)

❓ What is Adam optimization? How do you implement it?

Adam = Momentum + RMSProp combined.

Keeps moving average of gradients (m)

Keeps moving average of squared gradients (v)

Corrects them for bias

Rule:

m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * (gradient**2)

weights = weights - alpha * m / (sqrt(v) + epsilon)


Adam is the most popular optimizer today.

❓ What is learning rate decay? How do you implement it?

Instead of a fixed learning rate, decrease it slowly as training goes on:

Why?

Large steps early (explore)

Small steps later (fine-tune)

Formula example:

alpha_t = alpha0 / (1 + decay_rate * t)


t = iteration number

❓ What is batch normalization? How do you implement it?

Batch Normalization = normalize inside the network.

Normalizes each layer’s outputs to mean 0 and variance 1 per mini-batch.

Then adds two learnable parameters (γ and β) to allow scaling/shifting.

Why?

Training is faster

Training is more stable

Allows higher learning rates

✅ Summary

Hyperparameter = training knob

Normalization = keep features similar

Saddle point = flat point, not min

SGD = update per sample

Mini-batch = update per group

Moving average = smooth sequence

Momentum = adds speed

RMSProp = scales by squared gradients

Adam = Momentum + RMSProp

LR decay = shrink step size over time

Batch norm = normalize inside layers