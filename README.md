# Chatbot seq2seq

## Introduction

Use the [**Chatbot**](https://github.com/tp6fu6m3/Chatbot) to train a Sequence to Sequence model.

## Quick Start

1. Clone and navigate to the downloaded repository. Further, install required pip packages.

```
git clone https://github.com/tp6fu6m3/Chatbot_seq2seq.git
cd Chatbot_seq2seq
pip3 install -r requirements.txt
```

2. Download [**Chatbot**](https://github.com/tp6fu6m3/Chatbot)

```
cd data
git clone https://github.com/tp6fu6m3/Chatbot.git
cd ..
```

3. Train the model and save it as `model.h5`.

```
python3 train.py
```

4. Demonstrate the chatbot with the well-trained model.

```
python3 demo.py
```
