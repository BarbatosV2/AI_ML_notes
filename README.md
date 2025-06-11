# AI/ML notes by Zaw

# 🔁 Common Layer Patterns

🧱 Pattern 1: MLP (Multi-Layer Perceptron)
```
Input → Linear → ReLU → Linear → ReLU → Linear → Softmax
```
🧱 Pattern 2: With Regularization
```
Input → Linear → BatchNorm → ReLU → Dropout → Linear → ReLU → Output
```
📷 Pattern 3: CNN Block
```
Input Image
  ↓
Conv2D → BatchNorm2D → ReLU → MaxPool
  ↓
Repeat...
  ↓
Flatten → Dense → Output
```
📖 Pattern 4: Transformer Block
```
Input Embeddings
  ↓
Self-Attention → Add & Norm → Feedforward → Add & Norm
```


# 🚫 Examples of Bad/Invalid Layer Orderings

❌ ReLU → Linear: Non-linearity before linear transformation breaks pattern.

❌ Softmax in hidden layers: Use softmax only at the output layer for classification.

❌ Dropout before BatchNorm: Dropout may ruin the statistics needed by BatchNorm.

❌ BatchNorm after output layer: Output layer should produce raw logits or class probs without normalization.
