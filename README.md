# JSMA — Jacobian-based Saliency Map Attack

This is an implementation of the paper **"The Limitations of Deep Learning in Adversarial Settings"** by Papernot et al. (IEEE EuroS&P 2016) — [arXiv:1511.07528](https://arxiv.org/abs/1511.07528)

The basic idea of the paper: if you know a neural network's weights, you can figure out exactly which pixels to nudge in an image so the network misclassifies it — while the image still looks completely normal to a human.

---

## What's in each file

- `model.py` — the LeNet-5 network, built to match the paper exactly
- `jsma.py` — the actual attack: Jacobian computation, saliency maps, the crafting loop
- `train.py` — trains LeNet on MNIST
- `attack.py` — runs the attack on a batch of samples and collects results
- `evaluate.py` — computes the hardness and adversarial distance metrics from the paper
- `visualize.py` — generates the figures from the paper
- `demo.py` — a simple end-to-end script to see everything working together
- `utils.py` — small helper functions used across the project

---

## Getting started

```bash
pip install torch torchvision numpy matplotlib tqdm

# train the model first (about 5 min on CPU)
python train.py

# then run the demo to see the attack in action
python demo.py

# or run the full evaluation to reproduce the paper numbers
python attack.py --n_samples 100 --max_distortion 0.145 --theta 1.0
```

---

## What the paper achieves (and what this code reproduces)

The attack succeeds **97.10% of the time** while only modifying around **4% of pixels** per image — on average just 32 pixels out of 784. And humans shown the modified images still correctly identify the digit.

---

## How the attack works (plain English)

1. Do a forward pass through the network and compute the Jacobian — basically asking "if I change pixel i, how much does each output class change?"
2. Use that information to build a saliency map — a ranking of which pixels are most useful to perturb for a given target class
3. Pick the top two pixels, nudge them, and repeat
4. Stop when the network predicts the target class, or you've modified too many pixels

The math behind step 1 (the saliency map condition):
```
only perturb pixel i if:
  - it increases the target class output
  - it decreases all other class outputs at the same time
```

---

## Network architecture

```
Input (28×28 image, flattened to 784 values)
  → Conv layer (20 filters, 5×5)  + ReLU + MaxPool
  → Conv layer (50 filters, 5×5)  + ReLU + MaxPool
  → Fully connected (500 neurons) + ReLU
  → Fully connected (10 neurons)  ← Jacobian is computed here, not after softmax
  → Softmax                       ← used for final predictions only
```

One thing worth noting: the Jacobian is computed on the layer *before* softmax. The paper points out that softmax produces extreme gradient values that mess up the saliency maps, so using the raw logits gives much better results.

---

## Project structure

```
jsma_attack/
├── README.md
├── model.py
├── jsma.py
├── train.py
├── attack.py
├── evaluate.py
├── visualize.py
├── demo.py
├── utils.py
└── checkpoints/
    └── lenet_mnist.pth   (created after running train.py)
```