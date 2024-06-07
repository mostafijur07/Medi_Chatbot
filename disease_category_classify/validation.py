import torch
import torchmetrics

def run_validation(model, validation_ds, tokenizer_src, max_len, device, print_msg, global_step, writer, num_examples):
    model.eval() # Set the model to evaluation mode
    console_width = 80

    with torch.no_grad(): # Disable gradient calculations
        count = 0
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            label = batch["label"].to(device)
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # Pass through the model
            encoder_output = model.encode(encoder_input, encoder_mask)
            proj_output = model.project(encoder_output)  # Classification token
            predicted_class = torch.argmax(proj_output, dim=-1).squeeze() #Used torch.argmax to get the predicted class index.

            source_text = batch["src_text"][0]
            target_label = label.item()
            predicted_label = predicted_class.item()

            # Label mapping for display
            label_map = {0: "skin related", 1: "fever related", 2:"gastric related", 3:"others query"}
            target_text = label_map[target_label]
            predicted_text = label_map[predicted_label]

            # Print source, target, and predicted texts
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{predicted_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break