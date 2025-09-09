# Machine Learning – Beginner Optimization Concepts

This document explains some common machine learning concepts and optimizers in a **beginner-friendly way**.  
It is written in a simple **Q&A style** so you can quickly find the answers you need.

---

## ❓ What is a hyperparameter?
A **hyperparameter** is like a *setting knob* you choose **before training** a model.  
They are not learned from the data; instead, they control how the training happens.  

**Examples:**
- Learning rate (size of the update steps in gradient descent)  
- Number of hidden layers in a neural network  
- Batch size during training  

---

## ❓ How and why do you normalize your input data?
**Normalize = put all features on a similar scale.**  

**Why?**  
If one feature (like age = 20–80) and another (like income = 1000–100000) have very different ranges, the model struggles.  

**How:**
- **Min–max scaling:**  
  \[
  x' = \frac{x - \min(x)}{\max(x) - \min(x)}
  \]  
- **Standardization (z-score):**  
  \[
  x' = \frac{x - \mu}{\sigma}
  \]  

This makes training **faster** and **more stable**.  

---

## ❓ What is a saddle point?
A **saddle point** is like the seat of a horse saddle:  
- Going forward/back, it curves down (like a valley).  
- Going left/right, it curves up (like a hill).  

It’s not a true minimum, but gradient descent can get stuck there because the slope looks flat.  

---

## ❓ What is stochastic gradient descent (SGD)?
Gradient descent normally computes gradients using the **whole dataset** (slow).  

**Stochastic Gradient Descent (SGD):**  
- Uses **just 1 example** at a time.  
- Much faster but noisy.  

---

## ❓ What is mini-batch gradient descent?
A compromise between full batch and SGD:  
- Uses **small groups of samples** (mini-batches) like 32 or 128 examples.  
- Faster than full batch, less noisy than pure SGD.  
- The most common approach in practice.  

---

## ❓ What is a moving average? How do you implement it?
A **moving average** smooths out values over time.  

**Example:** Loss values `[5, 4, 6, 3]` with window = 2  
→ Moving averages = `[4.5, 5, 4.5]`  

**Python:**
```python
def moving_average(values, window):
    ma = []
    for i in range(len(values) - window + 1):
        ma.append(sum(values[i:i+window]) / window)
    return ma
```
---

## ❓ What is gradient descent with momentum?
Think of a ball rolling down a hill. Instead of stopping at every bump, it **builds speed** by remembering past directions (gradients). This helps it move faster and not get stuck easily.

**Update rule:**
```text
v = beta * v + (1 - beta) * gradient
weights = weights - alpha * v
```

---

## ❓ What is RMSProp? How do you implement it?

RMSProp scales each parameter’s step by a moving average of its squared gradients.  
This keeps steps stable even if some gradients are large or noisy.

**Update rule:**
```python
s = beta * s + (1 - beta) * (gradient**2)
weights = weights - alpha * gradient / (sqrt(s) + epsilon)
```

- `s` = running average of squared gradients  
- `beta ≈ 0.9`  
- `epsilon ~ 1e-8` (to avoid division by zero)

---

## ❓ What is Adam optimization? How do you implement it?

Adam = Momentum + RMSProp.  

It keeps track of:
- `m` = moving average of gradients (momentum)  
- `v` = moving average of squared gradients (RMSProp part)  

Adam uses **bias correction** so early estimates aren’t too small.

**Update rule (simplified):**
```python
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * (gradient**2)

m_hat = m / (1 - beta1**t)   # bias-corrected
v_hat = v / (1 - beta2**t)

weights = weights - alpha * m_hat / (sqrt(v_hat) + epsilon)
```

Typical defaults:  
- `beta1 = 0.9`  
- `beta2 = 0.999`  
- `epsilon = 1e-8`

---

## ❓ What is learning rate decay? How do you implement it?

Instead of using a fixed learning rate, decrease it over time:  
- **Big steps early** → explore  
- **Smaller steps later** → fine-tune  

**Example schedules:**
```python
# Inverse time decay
alpha_t = alpha0 / (1 + decay_rate * t)

# Step decay (every k steps)
alpha_t = alpha0 * (gamma ** floor(t / k))

# Cosine decay (popular for deep nets)
alpha_t = 0.5 * alpha0 * (1 + cos(pi * t / T_max))
```

- `t` = iteration or epoch number

---

## ❓ What is batch normalization? How do you implement it?

Batch Normalization normalizes layer outputs within each mini-batch to mean 0 and variance 1,  
then learns a **scale (`gamma`)** and **shift (`beta`)**.  

This stabilizes and speeds up training, and often allows larger learning rates.

**Forward pass (per feature/channel):**
```python
mu    = mean(a, batch)
var   = mean((a - mu)**2, batch)
a_norm = (a - mu) / sqrt(var + epsilon)
out    = gamma * a_norm + beta
```

During training, keep running `mu` and `var` for inference.

---

## ✅ Summary

- **Momentum**: adds velocity → faster, smoother updates  
- **RMSProp**: scales steps by squared-gradient average  
- **Adam**: Momentum + RMSProp with bias correction (great default)  
- **Learning Rate Decay**: shrink learning rate over time  
- **Batch Normalization**: normalize activations per batch, then scale/shift  
