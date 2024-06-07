import torch
from torch.utils.data import Dataset  #The base class for all PyTorch datasets.

class MediAndNonMediQueryDataset(Dataset):
    def __init__(self, ds, tokenizer, input_key, output_key, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer = tokenizer
        self.input_key = input_key
        self.output_key = output_key

        self.pad_token = torch.tensor([tokenizer.token_to_id("[PAD]")], dtype=torch.int64) #Creates a tensor for the padding token.
        self.cls_token = torch.tensor([tokenizer.token_to_id("[CLS]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds) # Returns: The length of the dataset.
    
    def __getitem__(self, index):
        data = self.ds[index]
        src_text = data['Category'][self.input_key]
        tgt_label = data['Category'][self.output_key]

        # Tokenize the input text
        enc_input_tokens = self.tokenizer.encode(src_text).ids

        # Calculate the number of padding tokens needed
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 1

        if enc_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Prepare the encoder input with padding
        encoder_input = torch.cat(
            [
                self.cls_token, #CLS token
                torch.tensor(enc_input_tokens, dtype=torch.int64), #input_sentance
                self.pad_token.repeat(enc_num_padding_tokens), #padding_token.
            ],
            dim=0,
        )

        # Create the encoder mask
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() #Use unsqueeze(0) two time to add seq_len and batch dimentation.

        # Map the output label to a numeric value
        label_map = {"skin related": 0, "fever related": 1, "gastric related": 2, "others query": 3}
        label = torch.tensor(label_map[tgt_label], dtype=torch.int64) 

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, #The tokenized and padded input text.
            "encoder_mask": encoder_mask, #A mask indicating the non-padding parts of the input.
            "label": label, #The numeric label for the classification task.
            "src_text": src_text #The original input text.
        }