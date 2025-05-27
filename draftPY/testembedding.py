import numpy as np

file_path = "etp_corpus_A_embeddings.npz"
with np.load(file_path, allow_pickle=True) as data:
    print("Keys in the NPZ file:")
    for key in data.files:
        print(f"- {key}, Shape: {data[key].shape if hasattr(data[key], 'shape') else 'N/A (scalar or object)'}, Dtype: {data[key].dtype if hasattr(data[key], 'dtype') else 'N/A'}")

    print("\n--- Metadata ---")
    if 'metadata' in data.files:
        # The metadata was saved as a 0-d object array containing a JSON string
        metadata_str = data['metadata'].item() # .item() to get the string from the 0-d array
        import json
        metadata_dict = json.loads(metadata_str)
        for k, v in metadata_dict.items():
            print(f"  {k}: {v}")
    else:
        print("Metadata key not found.")

    # Example: Load the hidden state of the first item from layer 0
    if 'hidden_state_layer_0_item_0' in data.files:
        hs_layer0_item0 = data['hidden_state_layer_0_item_0']
        print(f"\nShape of hidden_state_layer_0_item_0: {hs_layer0_item0.shape}") # Should be (seq_len, hidden_size)
        # For dummy texts with max_length 512 and hidden_size of Qwen1.5B (e.g. 2048 for 1.8B)
        # this should be (512, 2048) or similar.

    # Example: Load the attention of the first item from layer 0
    if 'attentions_attention_layer_0_item_0' in data.files:
        attn_layer0_item0 = data['attentions_attention_layer_0_item_0']
        print(f"\nShape of attentions_attention_layer_0_item_0: {attn_layer0_item0.shape}") # Should be (num_heads, seq_len, seq_len)
        # E.g., (num_heads, 512, 512)