import os
from collections import OrderedDict
import numpy as np
import torch
from tokenizers import Tokenizer
import re

from torch import nn

cfg = {
    'vocab_size': 20000,
    'emb_dim': 64,
    'hidden_dim': 64,
    'num_layers': 3,
    'bidirectional': True,
    'dropout': 0.6,
    'seq_len': 256,
}


class SentimentAnalysisModel(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super(SentimentAnalysisModel, self).__init__()

        # input : (batch_size , sequence_length) ---> (64,256) in training
        self.embedding = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        # output : (batch_size , sequence_length , embedding_dim) every token get it's embedding

        # input : (batch_size , sequence_length , embedding_dim)
        self.lstm = nn.LSTM(
            input_size=cfg['emb_dim'],
            hidden_size=cfg['hidden_dim'],
            num_layers=cfg['num_layers'],
            batch_first=True,
            bidirectional=cfg['bidirectional'],
            dropout=cfg['dropout']
        )
        # output : (batch_size , sequence_length , hidden_dim*2) because it's bidirectinoal

        # conv1d expects the shape of input as following
        # (batch_size , hidden_dim*2 , sequence_length) so we need to convert the last output to this

        # input (batch_size , sequence_length , hidden_dim*2)
        # -> !(batch_size , hidden_dim*2 , sequence_length)! (used permute in forward)
        self.conv1 = nn.Conv1d(
            cfg['hidden_dim'] * (2 if cfg['bidirectional'] else 1), 128, kernel_size=3, padding=1
        )
        # output: (batch_size,out_channels,sequence_length)

        # input: (batch_size,out_channels,sequence_length)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # output: (batch_size,out_channels , sequence_length/2)

        # input: (batch_size,out_channels , sequence_length/2) --> (batch_size,out_channels*sequence_length/2)
        self.fc1 = nn.Linear(128 * (cfg['seq_len'] // 2), 64)
        # output: (batch_size , 64)

        # input: (batch_size,64)
        self.fc2 = nn.Linear(64, 10)
        # output: 10
        self.dropout = nn.Dropout(cfg['dropout'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)

        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(0, 2, 1)

        conv_out = self.conv1(lstm_out)
        pooled_out = self.pool(conv_out)

        flattened = pooled_out.view(pooled_out.size(0), -1)

        x = self.dropout(torch.relu(self.fc1(flattened)))
        logits = self.fc2(x)

        return logits


def get_model(model_path: str) -> SentimentAnalysisModel:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loaded_model = SentimentAnalysisModel(cfg)

    state_dict = torch.load(model_path, map_location=device)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    loaded_model.load_state_dict(new_state_dict)
    loaded_model = loaded_model.to(device)
    loaded_model.eval()

    return loaded_model


def get_tokenizer(tokenizer_path: str) -> Tokenizer:
    return Tokenizer.from_file(tokenizer_path)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.strip()
    return text


def inference(text: str, model: SentimentAnalysisModel, tokenizer: Tokenizer, device: torch.device) -> tuple[
    int, np.ndarray]:
    model.eval()
    cleaned_text = clean_text(text)
    encoded = tokenizer.encode(cleaned_text)
    input_ids = torch.tensor([encoded.ids]).to(device)

    with torch.no_grad():
        output = model(input_ids)
        probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

    predicted_class = torch.argmax(output, dim=1).item()
    rating = predicted_class + 1

    return rating, probabilities


def get_prediction(text: str) -> int:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'sentiment_analysis_model.pth')
    tokenizer_path = os.path.join(current_dir, 'bpe_tokenizer.json')

    res, _ = inference(
        text,
        get_model(model_path),
        get_tokenizer(tokenizer_path),
        torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return res


if __name__ == '__main__':
    loaded_model = get_model('sentiment_analysis_model.pth')
    tokenizer = get_tokenizer('bpe_tokenizer.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    text_to_analyze = 'what a very bad movie'
    rating, probability = inference(text_to_analyze, loaded_model, tokenizer, device)
    print(f'Predicted Rating: {rating}')
    print(f'Probability: {probability[rating - 1]:.4f}')

    text_to_analyze = 'This movie was absolutely incredible, a must-watch!'
    rating, probabilities = inference(text_to_analyze, loaded_model, tokenizer, device)
    print(f'Predicted Rating: {rating}')
    print(f'Class Probabilities: {["{:.4f}".format(p) for p in probabilities]}')

    text_to_analyze = 'Mid movie, would not recommend, but i would watch it again'
    rating, probabilities = inference(text_to_analyze, loaded_model, tokenizer, device)
    print(f'Predicted Rating: {rating}')
    print(f'Class Probabilities: {["{:.4f}".format(p) for p in probabilities]}')

    print(get_prediction('Was not impressed by the movie'))
