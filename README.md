# Chatbot implemented by sequence to sequence LSTM

This repository contains a keras implementation of sequence to sequence LSTM model running on the your line conversation dataset augmented by [**Chatbot**](https://github.com/tp6fu6m3/Chatbot).

## Quick Start

1. Clone and navigate to the downloaded repository. Further, install required pip packages.

```
git clone https://github.com/tp6fu6m3/Chatbot_seq2seq.git
cd Chatbot_seq2seq
pip3 install -r requirements.txt
```

2. Download [**Chatbot**](https://github.com/tp6fu6m3/Chatbot) to conduct data augmentation with our line conversation dataset.

```
cd data
git clone https://github.com/tp6fu6m3/Chatbot.git
cd ..
```

3. Prepare your own line conversation record under `data/` as `history.txt` or use my provided history.

![Imgur](https://briian.com/wp-content/uploads/2013/03/LINE-backup-001.png)

4. Train the model and save it as `model.h5`. We do the data preprocessing by using `history.txt` and `Chatbot` to generate `messageList.txt` and `replyList.txt` for training.

```
python3 train.py
```

5. Demonstrate the chatbot with the well-trained model.

```
python3 demo.py
```

## Repository Structure

```
Chatbot_seq2seq
├── data
│   ├── history.txt
│   ├── messageList.txt
│   ├── replyList.txt
│   └── reader.py
├── model
│   ├── model.h5
│   └── model.py
├── demo.py
├── README.md
├── requirements.txt
├── train.py
└── utils.py

```
