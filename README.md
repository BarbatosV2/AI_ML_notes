# AI/ML note by Zaw

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

# ✅ Summary Table: What Goes After What

| Layer |	Usually Followed By |
| --- | --- |
| Linear |	BatchNorm or Activation |
| BatchNorm |	Activation |
| Activation	| Dropout or Linear |
| Dropout	| Linear |
| Conv2D	| BatchNorm → Activation → Pool |
| Recurrent	| Dropout or another RNN block |
| Output	| Sigmoid (binary) or Softmax (multiclass) |



## 🧠 1. Activation Functions

These make neural networks non-linear (learn complex patterns).

Name	Use Case	Output Range
ReLU	Default for hidden layers	[0, ∞)
Sigmoid	Binary classification	[0, 1]
Tanh	Balanced activations	[-1, 1]
Softmax	Multiclass classification	Probabilities summing to 1

👉 Tip: Use ReLU for hidden layers, and only use sigmoid/softmax at the output layer if needed.

## 🔢 2. Loss Functions

They guide learning — you minimize these during training.

Task Type	Loss Function
Regression	MSE (Mean Squared Error)
Binary Classification	BCE (Binary Cross Entropy)
Multiclass Classification	CrossEntropyLoss

## 🧮 3. Weight Initialization

Proper weight init helps training converge faster. PyTorch/TF does this for you automatically with methods like:

Xavier (Glorot) initialization

He initialization (for ReLU)

## 🧪 4. Overfitting & Underfitting

Problem	Sign	Fixes
Overfitting	Train accuracy >> test accuracy	Add dropout, regularization, more data
Underfitting	Both accuracies low	Bigger model, longer training, tune LR

## 🚀 5. Optimizer Choice

Optimizers adjust weights during training:

Optimizer	Strengths
SGD	Simple, but slow convergence
Adam	Most commonly used, fast & adaptive
RMSProp	Used in RNNs

👉 Adam is a good default.

## 🧪 6. Learning Rate

Too small = slow learning
Too big = unstable training
✅ Tip: use learning rate scheduling or try lr_find in tools like PyTorch Lightning or FastAI.

## ⏳ 7. Epochs, Batches, and Steps

Batch: Subset of data processed at once (affects speed and generalization)

Epoch: One full pass over the dataset

Steps: Batches per epoch

## 📊 8. Metrics

Use the right metrics to evaluate your model:

Task	Metrics
Classification	Accuracy, Precision, Recall, F1
Regression	MSE, MAE, R²

## 🧱 9. Model Types to Learn About

MLP (for tabular data)

CNN (images)

RNN / LSTM / GRU (sequences)

Transformer / Attention (modern NLP and vision)

## ⚙️ 10. Frameworks to Know

| Library | Use Case | 
| --- | --- |
| PyTorch |	Research & flexibility |
| TensorFlow | Production & scalability |
| Keras	| Beginner-friendly (TF-based) |
| JAX	| Fast gradient computation (research/ML systems) |

## 💡 Bonus: Tips
Use GPU (CUDA) for faster training.

Use early stopping to avoid overfitting.

Use normalization on your input data (standard scaling for MLPs, mean/std for images).

Visualize training with TensorBoard or Weights & Biases (wandb).

Save models with .pt or .pth (torch.save()), and load them with torch.load().


