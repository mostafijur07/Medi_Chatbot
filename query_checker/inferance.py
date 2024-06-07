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
        label_map = {0: "medical query", 1: "non-medical query"}
        predicted_label = label_map[predicted_class.item()]
        
        return predicted_label
    
# Define the device
device = torch.device("cpu")
config = get_config()
model, tokenizer = load_model_and_tokenizer(config)

@app.route('/predict_query_checker', methods=['POST'])
def predict():
    data = request.json
    input_sentence = data['input_sentence']
    predicted_label = input_to_output(input_sentence, model, tokenizer, device)
    print(predicted_label)
    return jsonify({'predicted_label': predicted_label})

if __name__ == '__main__':
    app.run(debug=True, port=5000)


# INFERENCE USING VALIDATION DATASET:
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
#     train_dataloader, val_dataloader, tokenizer_src = get_ds(config)
#     model = get_model(config, tokenizer_src.get_vocab_size(), config['num_classes']).to(device)

#     # Load the pretrained weights
#     model_filename = latest_weights_file_path(config)
#     state = torch.load(model_filename)
#     model.load_state_dict(state['model_state_dict'])

#     run_validation(model, val_dataloader, tokenizer_src, config['seq_len'], device, lambda msg: print(msg), 0, None, num_examples=10)

# if __name__ == "__main__":
#     main()


## MANUAL INFERANCE CODE:

# from pathlib import Path
# import torch
# import torch.nn as nn
# from config import get_config, latest_weights_file_path
# from train import get_model, get_ds, run_validation
# from tokenizers import Tokenizer

# def load_model_and_tokenizer(config):
#     device = torch.device("cpu")
#     tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format('en'))))
#     # tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format('tgt'))))
#     model = get_model(config, tokenizer_src.get_vocab_size(), config['num_classes']).to(device)  # Use num_classes from config
#     # Load the pretrained weights
#     model_filename = latest_weights_file_path(config)
#     state = torch.load(model_filename)
#     model.load_state_dict(state['model_state_dict'])
#     return model, tokenizer_src

# def input_to_output(input_sentence, model, tokenizer_src, config, device="cpu"):
#     model.eval()
#     with torch.no_grad():
#         input_ids = tokenizer_src.encode(input_sentence).ids
#         input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # Add batch dimension
#         encoder_mask = (input_tensor != tokenizer_src.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(1)  # Create mask for encoder
#         encoder_output = model.encode(input_tensor, encoder_mask)
#         proj_output = model.project(encoder_output)  # Classification token
#         predicted_class = torch.argmax(proj_output, dim=-1).squeeze()  # Get the predicted class index
        
#         # Label mapping for display
#         label_map = {0: "medical query", 1: "non-medical query"}
#         predicted_label = label_map[predicted_class.item()]
        
#         return predicted_label

# def main():
#     input_sentence = "i have continuous nose pain."
#     config=get_config()
#     model, tokenizer = load_model_and_tokenizer(config)
#     output_sentence = input_to_output(input_sentence, model, tokenizer, config)

#     print("Input Sentence:", input_sentence)
#     print("Output Sentence:", output_sentence)

# if __name__ == "__main__":
#     main()