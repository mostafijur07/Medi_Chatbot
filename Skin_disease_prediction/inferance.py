#SERVER CODE 

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import torch
import torch.nn as nn
from config import get_config, latest_weights_file_path
from train import get_model, get_ds
from tokenizers import Tokenizer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})

def load_model_and_tokenizer(config):
    device = torch.device("cpu")
    tokenizer = Tokenizer.from_file(str(Path('tokenizer_en.json')))
    model = get_model(config, tokenizer.get_vocab_size(), config['num_classes']).to(device)  # Use num_classes from config
    # Load the pretrained weights
    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])
    return model, tokenizer

def input_to_output(input_sentence, model, tokenizer, device):
    model.eval()
    with torch.no_grad():
        input_ids = tokenizer.encode(input_sentence).ids
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension
        encoder_mask = (input_tensor != tokenizer.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(1)  # Create mask for encoder
        encoder_output = model.encode(input_tensor, encoder_mask)
        proj_output = model.project(encoder_output)  # Classification token
        predicted_class = torch.argmax(proj_output, dim=-1).squeeze()  # Get the predicted class index
        
        # Label mapping for display
        train_dataloader, val_dataloader, tokenizer, num_classes, train_ds = get_ds(config)
        label_map = val_dataloader.dataset.label_map
        inverted_label_map = {value: key for key, value in label_map.items()}
        predicted_label = inverted_label_map[predicted_class.item()]
        
        return predicted_label
    
# Define the device
device = torch.device("cpu")
config = get_config()
model, tokenizer = load_model_and_tokenizer(config)

@app.route('/predict_skin_disease', methods=['POST'])
def predict():
    data = request.json
    input_sentence = data['input_sentence']
    predicted_disease = input_to_output(input_sentence, model, tokenizer, device)
    print(predicted_disease)
    return jsonify({'predicted_disease': predicted_disease})

if __name__ == '__main__':
    app.run(debug=True, port=5002)


# from pathlib import Path
# import torch
# import torch.nn as nn
# from config import get_config, latest_weights_file_path
# from train import get_model, get_ds, run_validation
# from tokenizers import Tokenizer

# def main():
#     # Define the device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)
#     config = get_config()
#     train_dataloader, val_dataloader, tokenizer_src, num_classes, train_ds = get_ds(config)
#     model = get_model(config, tokenizer_src.get_vocab_size(), num_classes).to(device)

#     # Load the pretrained weights
#     model_filename = latest_weights_file_path(config)
#     state = torch.load(model_filename)
#     model.load_state_dict(state['model_state_dict'])

#     run_validation(model, train_ds, val_dataloader, tokenizer_src, config['seq_len'], device, lambda msg: print(msg), 0, None, num_examples=10)

# if __name__ == "__main__":
#     main()