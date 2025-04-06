# --- Imports ---
import os
import json
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import re
from flask import Flask, request, jsonify
from huggingface_hub import hf_hub_download
import logging
import time

# --- Configuration ---
HF_REPO_ID = "rxmha125/my-scratch-transformer-colab"
DEVICE = "cpu"

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("API Service Starting...")
logging.info(f"Loading model from Hugging Face Hub: {HF_REPO_ID}")
logging.info(f"Using device: {DEVICE}")

# --- Global Variables for Model and Vocab (Load Once) ---
model = None
vocab_stoi = None
vocab_itos = None
config = None
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
UNK_IDX = 0
VOCAB_SIZE = None

# --- 1. Function to Load Artifacts ---
# [ Keep the load_artifacts function exactly as it was ]
def load_artifacts():
    """Downloads and loads model, config, and vocab."""
    global model, vocab_stoi, vocab_itos, config, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX, VOCAB_SIZE

    logging.info("Downloading artifacts from Hub...")
    try:
        config_path = hf_hub_download(repo_id=HF_REPO_ID, filename="config.json")
        vocab_stoi_path = hf_hub_download(repo_id=HF_REPO_ID, filename="vocab_stoi.json")
        vocab_itos_path = hf_hub_download(repo_id=HF_REPO_ID, filename="vocab_itos.json")
        model_weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename="pytorch_model.bin")
        logging.info("Downloads complete.")
    except Exception as e:
        logging.error(f"FATAL: Error downloading files from Hugging Face Hub: {e}", exc_info=True)
        raise SystemExit(f"Could not download model artifacts from {HF_REPO_ID}")

    logging.info("Loading configuration and vocab...")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        with open(vocab_stoi_path, 'r') as f:
            vocab_stoi = json.load(f)
        with open(vocab_itos_path, 'r') as f:
            vocab_itos_str = json.load(f)
            vocab_itos = {int(k): v for k, v in vocab_itos_str.items()}

        PAD_IDX = config.get('pad_idx', 1)
        SOS_IDX = config.get('sos_idx', 2)
        EOS_IDX = config.get('eos_idx', 3)
        UNK_IDX = config.get('unk_idx', 0)
        VOCAB_SIZE = config['vocab_size']
        logging.info(f"  Loaded vocab size: {VOCAB_SIZE}")
        logging.info(f"  PAD={PAD_IDX}, SOS={SOS_IDX}, EOS={EOS_IDX}, UNK={UNK_IDX}")
    except Exception as e:
        logging.error(f"FATAL: Error loading config/vocab JSON files: {e}", exc_info=True)
        raise SystemExit("Could not load config/vocab files.")

    # --- 3. Define Model Architecture (Keep as is) ---
    logging.info("Defining model architecture...")
    class PositionalEncoding(nn.Module):
        def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
            super(PositionalEncoding, self).__init__()
            den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
            pos = torch.arange(0, maxlen).reshape(maxlen, 1)
            pos_embedding = torch.zeros((maxlen, emb_size))
            pos_embedding[:, 0::2] = torch.sin(pos * den)
            pos_embedding[:, 1::2] = torch.cos(pos * den)
            pos_embedding = pos_embedding.unsqueeze(-2)
            self.dropout = nn.Dropout(dropout)
            self.register_buffer('pe', pos_embedding)

        def forward(self, token_embedding: torch.Tensor):
            token_embedding = token_embedding.to(self.pe.device)
            return self.dropout(token_embedding + self.pe[:token_embedding.size(0), :])

    class TokenEmbedding(nn.Module):
        def __init__(self, vocab_size: int, emb_size: int):
            super(TokenEmbedding, self).__init__()
            self.embedding = nn.Embedding(vocab_size, emb_size)
            self.emb_size = emb_size

        def forward(self, tokens: torch.Tensor):
            return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

    class Seq2SeqTransformer(nn.Module):
        def __init__(self,
                     num_encoder_layers: int,
                     num_decoder_layers: int,
                     emb_size: int,
                     nhead: int,
                     src_vocab_size: int,
                     tgt_vocab_size: int,
                     dim_feedforward: int,
                     dropout: float):
            super(Seq2SeqTransformer, self).__init__()
            self.transformer = Transformer(d_model=emb_size,
                                           nhead=nhead,
                                           num_encoder_layers=num_encoder_layers,
                                           num_decoder_layers=num_decoder_layers,
                                           dim_feedforward=dim_feedforward,
                                           dropout=dropout,
                                           batch_first=False)
            self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
            self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
            self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
            self.generator = nn.Linear(emb_size, tgt_vocab_size)

        def forward(self, src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
            src_emb = self.positional_encoding(self.src_tok_emb(src))
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
            outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                    src_key_padding_mask=src_padding_mask,
                                    tgt_key_padding_mask=tgt_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)
            return self.generator(outs)

        def encode(self, src: torch.Tensor, src_mask: torch.Tensor, src_padding_mask: torch.Tensor):
            src_emb = self.positional_encoding(self.src_tok_emb(src))
            return self.transformer.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)

        def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor, tgt_padding_mask: torch.Tensor, memory_key_padding_mask: torch.Tensor):
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
            return self.transformer.decoder(tgt_emb, memory,
                                            tgt_mask=tgt_mask, memory_mask=None,
                                            tgt_key_padding_mask=tgt_padding_mask,
                                            memory_key_padding_mask=memory_key_padding_mask)
    logging.info("Model architecture defined.")


    # --- 4. Instantiate Model and Load Weights ---
    # [ Keep the model instantiation and loading exactly as it was ]
    logging.info("Instantiating model...")
    try:
        model = Seq2SeqTransformer(
            num_encoder_layers=config.get('num_encoder_layers', 3),
            num_decoder_layers=config.get('num_decoder_layers', 3),
            emb_size=config.get('emb_size', 256),
            nhead=config.get('nhead', 4),
            src_vocab_size=VOCAB_SIZE,
            tgt_vocab_size=VOCAB_SIZE,
            dim_feedforward=config.get('ffn_hid_dim', 512), # Check key in config.json
            dropout=config.get('dropout', 0.1)
        )
        logging.info("Model instantiated.")
        logging.info(f"Loading weights...")
        # Correct path variable was used before, ensure model_weights_path is defined in load_artifacts
        # Need to make model_weights_path accessible or re-fetch it if load_artifacts scope was limited
        # Assuming model_weights_path is accessible from load_artifacts via globals or return values
        # (The previous version implicitly relied on global scope access within load_artifacts)
        # Let's adjust load_artifacts to ensure paths are accessible if needed, although globals work here.
        model_weights_path = hf_hub_download(repo_id=HF_REPO_ID, filename="pytorch_model.bin") # Re-download here for clarity if not global

        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device(DEVICE)))
        model.to(DEVICE)
        model.eval()
        logging.info(f"Model weights loaded and model set to evaluation mode on {DEVICE}.")

    except Exception as e:
        logging.error(f"FATAL: Error instantiating model or loading weights: {e}", exc_info=True)
        raise SystemExit("Could not instantiate/load model.")


# --- 5. Helper Functions (Keep as is) ---
# [ Keep simple_tokenizer and generate_square_subsequent_mask exactly as they were ]
def simple_tokenizer(text):
    if not isinstance(text, str): text = str(text)
    text = text.lower()
    text = re.sub(r"([,.!?\"':;()])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def generate_square_subsequent_mask(sz: int, device: torch.device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# --- Generation Function (Keep as is) ---
# [ Keep generate_response exactly as it was ]
def generate_response(prompt: str, max_len: int = 50) -> str:
    global model, vocab_stoi, vocab_itos, config, DEVICE, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX

    if not all([model, vocab_stoi, vocab_itos, config]):
        logging.error("Model/vocab not loaded properly.")
        return "[Error: Model or vocab not initialized]"

    model.eval()

    prompt_tokens = simple_tokenizer(prompt)
    prompt_ids = [vocab_stoi.get(token, UNK_IDX) for token in prompt_tokens]

    if not all(0 <= token_id < VOCAB_SIZE for token_id in prompt_ids):
         logging.warning(f"Prompt contains tokens mapping to invalid IDs: {prompt_tokens} -> {prompt_ids}")
         return "[Error: Prompt contains invalid tokens]"

    try:
        src_tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(1).to(DEVICE)
    except Exception as e:
        logging.error(f"Error creating source tensor from IDs {prompt_ids}: {e}", exc_info=True)
        return "[Error during tensor creation]"

    num_tokens = src_tensor.shape[0]
    src_mask = torch.zeros((num_tokens, num_tokens), device=DEVICE).type(torch.bool)
    src_padding_mask = (src_tensor == PAD_IDX).transpose(0, 1)

    # --- Encoding ---
    try:
        with torch.no_grad():
            memory = model.encode(src_tensor, src_mask, src_padding_mask)
    except Exception as e:
        logging.error(f"Error during encoding: {e}", exc_info=True)
        return "[Error during encoding]"

    # --- Decoding Loop ---
    tgt_tokens = torch.tensor([[SOS_IDX]], dtype=torch.long, device=DEVICE)
    generated_ids = []

    try:
        for i in range(max_len - 1):
            tgt_seq_len = tgt_tokens.shape[0]
            tgt_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE)
            tgt_padding_mask = torch.zeros((tgt_tokens.shape[1], tgt_seq_len), device=DEVICE).type(torch.bool)
            memory_key_padding_mask = src_padding_mask

            with torch.no_grad():
                 decoder_output = model.decode(tgt_tokens, memory, tgt_mask,
                                               tgt_padding_mask=tgt_padding_mask,
                                               memory_key_padding_mask=memory_key_padding_mask)
                 logits = model.generator(decoder_output)
                 last_token_logits = logits[-1, :, :]
                 pred_token_id = last_token_logits.argmax(1).item()

            if not (0 <= pred_token_id < VOCAB_SIZE):
                 logging.error(f"Model predicted out-of-bounds token ID: {pred_token_id}. Vocab size is {VOCAB_SIZE}.")
                 break

            new_token_tensor = torch.tensor([[pred_token_id]], dtype=torch.long, device=DEVICE)
            tgt_tokens = torch.cat((tgt_tokens, new_token_tensor), dim=0)

            if pred_token_id == EOS_IDX:
                break
            else:
               generated_ids.append(pred_token_id)
        else:
             logging.info(f"Max generation length ({max_len}) reached.")

    except Exception as e:
        logging.error(f"Error during decode loop (iteration {i}): {e}", exc_info=True)
        return "[Error during decoding]"

    # --- Postprocessing ---
    generated_text = " ".join([vocab_itos.get(idx, "<unk>") for idx in generated_ids])
    generated_text = re.sub(r'\s+([,.!?])', r'\1', generated_text).strip()
    return generated_text

# --- Flask Application Setup ---
app = Flask(__name__)

# --- API Endpoint ---
@app.route('/api/generate', methods=['POST'])
def handle_generate():
    """API endpoint to generate text from a prompt."""
    if not request.is_json:
        logging.warning("Request received is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    prompt = data.get('prompt', None)
    max_length = data.get('max_length', 50)

    if not prompt:
        logging.warning("Request received with missing 'prompt' field")
        return jsonify({"error": "Missing 'prompt' field in JSON request"}), 400
    if not isinstance(prompt, str) or not prompt.strip():
         logging.warning(f"Invalid prompt received: '{prompt}'")
         return jsonify({"error": "Prompt must be a non-empty string"}), 400
    if not isinstance(max_length, int) or max_length <= 0:
        logging.warning(f"Invalid max_length received: {max_length}. Using default 50.")
        max_length = 50

    logging.info(f"Received prompt: '{prompt}' (max_length: {max_length})")

    try:
        start_time = time.perf_counter()
        response_text = generate_response(prompt, max_len=max_length)
        end_time = time.perf_counter()

        # <--- CHANGE: Calculate duration in minutes ---
        duration_seconds = end_time - start_time
        duration_minutes = duration_seconds / 60.0 # Convert seconds to minutes

        # <--- CHANGE: Update logging format ---
        logging.info(f"Generated response: '{response_text}' in {duration_minutes:.2f} min")

        if response_text.startswith("[Error:"):
            return jsonify({"error": response_text}), 500
        else:
            # <--- CHANGE: Modify successful response JSON key and value ---
            return jsonify({
                "response": response_text,
                "generation_time_min": round(duration_minutes, 2) # Use minutes rounded to 2 decimal places
            }), 200

    except Exception as e:
        logging.error(f"Unexpected error during generation/timing for prompt '{prompt}': {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred."}), 500

# --- Run the App ---
if __name__ == '__main__':
    try:
        load_artifacts()
        logging.info("Model and artifacts loaded successfully.")
        logging.info("Starting Flask development server...")
        app.run(debug=False, host='0.0.0.0', port=5001)
    except SystemExit as se:
        logging.error(f"Failed to initialize API: {se}")
    except Exception as e:
        logging.error(f"An unexpected error occurred before starting server: {e}", exc_info=True)
    finally:
        logging.info("API Service Shutting Down.")