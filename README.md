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

## 📚 Python Libraries and Their Use Cases with Examples

| Library            | Use Case                                            |
| ------------------ | --------------------------------------------------- |
| NumPy              | Numerical computing, array operations               |
| Pandas             | Data manipulation, analysis, DataFrames             |
| Matplotlib         | Data visualization (2D plots, graphs)               |
| Seaborn            | Statistical data visualization                      |
| Plotly             | Interactive plots and dashboards                    |
| Scikit-learn       | Machine learning models and tools                   |
| TensorFlow         | Deep learning, neural networks                      |
| PyTorch            | Deep learning, dynamic computational graphs         |
| Keras              | High-level neural networks API (on TensorFlow)      |
| OpenCV             | Computer vision and image processing                |
| Pillow (PIL)       | Image processing and manipulation                   |
| NLTK               | Natural Language Processing (NLP)                   |
| SpaCy              | Fast and efficient NLP                              |
| Transformers (HF)  | Pretrained LLMs, NLP (via HuggingFace)              |
| Requests           | Making HTTP requests                                |
| BeautifulSoup      | Web scraping (HTML parsing)                         |
| Scrapy             | Large-scale web scraping framework                  |
| SQLAlchemy         | Database interaction using ORM                      |
| PyMySQL / psycopg2 | MySQL / PostgreSQL database connectors              |
| Flask              | Lightweight web applications and APIs               |
| Django             | Full-stack web framework                            |
| FastAPI            | High-performance APIs with async support            |
| Jupyter Notebook   | Interactive coding and visualization                |
| Pygame             | Game development                                    |
| Bokeh              | Interactive visualizations (web-based)              |
| Dash               | Web dashboards for data apps                        |
| Pytest             | Unit testing                                        |
| Unittest           | Built-in unit testing framework                     |
| JAX                | High-performance ML with NumPy-like syntax          |
| Dask               | Parallel computing on large datasets                |
| LightGBM           | Gradient boosting (fast, efficient)                 |
| XGBoost            | Gradient boosting, commonly used in ML competitions |
| MLflow             | Machine learning lifecycle management               |

---

### ✨ Example Code for Each Library

#### NumPy

```python
import numpy as np
a = np.array([1, 2, 3])
print(a * 2)
```

#### Pandas

```python
import pandas as pd
df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [25, 30]})
print(df.describe())
```

#### Matplotlib

```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [4, 1, 9])
plt.title("Line Plot")
plt.show()
```

#### Seaborn

```python
import seaborn as sns
import pandas as pd
df = sns.load_dataset("iris")
sns.pairplot(df, hue="species")
```

#### Plotly

```python
import plotly.express as px
df = px.data.gapminder().query("year == 2007")
fig = px.scatter(df, x="gdpPercap", y="lifeExp", size="pop", color="continent")
fig.show()
```

#### Scikit-learn

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=1, noise=10)
model = LinearRegression().fit(X, y)
print(model.coef_)
```

#### TensorFlow

```python
import tensorflow as tf
x = tf.constant([[1.0, 2.0]])
dense = tf.keras.layers.Dense(units=1)
print(dense(x))
```

#### PyTorch

```python
import torch
import torch.nn as nn
x = torch.tensor([[1.0, 2.0]])
layer = nn.Linear(2, 1)
print(layer(x))
```

#### Keras

```python
from keras.models import Sequential
from keras.layers import Dense
model = Sequential([Dense(10, input_shape=(2,), activation='relu')])
print(model.summary())
```

#### OpenCV

```python
import cv2
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.jpg', gray)
```

#### Pillow (PIL)

```python
from PIL import Image
img = Image.open('image.jpg')
img_resized = img.resize((100, 100))
img_resized.save('resized.jpg')
```

#### NLTK

```python
import nltk
nltk.download('punkt')
print(nltk.word_tokenize("Hello, how are you?"))
```

#### SpaCy

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying a U.K. startup.")
for ent in doc.ents:
    print(ent.text, ent.label_)
```

#### Transformers (HF)

```python
from transformers import pipeline
qa = pipeline("question-answering")
print(qa(question="What is the capital of France?", context="Paris is the capital of France."))
```

#### Requests

```python
import requests
response = requests.get("https://api.github.com")
print(response.status_code)
```

#### BeautifulSoup

```python
from bs4 import BeautifulSoup
html = "<html><body><p>Hello</p></body></html>"
soup = BeautifulSoup(html, 'html.parser')
print(soup.p.text)
```

#### Scrapy

```python
# Run inside Scrapy project
import scrapy

class QuotesSpider(scrapy.Spider):
    name = "quotes"
    start_urls = ['http://quotes.toscrape.com']

    def parse(self, response):
        for quote in response.css('div.quote'):
            yield {'text': quote.css('span.text::text').get()}
```

#### SQLAlchemy

```python
from sqlalchemy import create_engine, Column, Integer, String, declarative_base
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    name = Column(String)

engine = create_engine('sqlite:///example.db')
Base.metadata.create_all(engine)
```

#### PyMySQL

```python
import pymysql
conn = pymysql.connect(host='localhost', user='user', password='pass', database='db')
cursor = conn.cursor()
cursor.execute("SELECT VERSION()")
print(cursor.fetchone())
```

#### Flask

```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, Flask!"

# app.run()  # Uncomment to run
```

#### Django

```bash
django-admin startproject mysite
cd mysite
python manage.py runserver
```

#### FastAPI

```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "FastAPI"}

# Run with: uvicorn filename:app --reload
```

#### Jupyter Notebook

```python
# Inside a Jupyter notebook cell
print("This is a Jupyter Notebook!")
```

#### Pygame

```python
import pygame
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Hello Pygame")
```

#### Bokeh

```python
from bokeh.plotting import figure, show
plot = figure()
plot.line([1, 2, 3], [4, 6, 2])
show(plot)
```

#### Dash

```python
from dash import Dash, html
app = Dash(__name__)
app.layout = html.Div("Hello Dash!")
# app.run_server()  # Uncomment to run
```

#### Pytest

```python
# test_sample.py
def add(x, y): return x + y

def test_add():
    assert add(2, 3) == 5
```

#### Unittest

```python
import unittest

class TestAdd(unittest.TestCase):
    def test_add(self):
        self.assertEqual(2 + 3, 5)

# if __name__ == '__main__':
#     unittest.main()
```

#### JAX

```python
import jax.numpy as jnp
from jax import grad

def f(x): return x ** 2
df = grad(f)
print(df(3.0))
```

#### Dask

```python
import dask.array as da
x = da.ones((10000, 10000), chunks=(1000, 1000))
print(x.mean().compute())
```

#### LightGBM

```python
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
lgbm = lgb.LGBMClassifier()
lgbm.fit(data.data, data.target)
```

#### XGBoost

```python
import xgboost as xgb
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
model = xgb.XGBClassifier()
model.fit(X, y)
```

#### MLflow

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("param1", 5)
    mlflow.log_metric("accuracy", 0.9)
```

