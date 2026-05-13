#!/usr/bin/env python3
"""Extract tokenizer vocab and merges from a GGUF file into binary files."""
import struct, sys, os

def extract_tokenizer(gguf_path, out_dir):
    with open(gguf_path, 'rb') as f:
        magic = f.read(4)
        ver = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<q', f.read(8))[0]
        n_kv = struct.unpack('<q', f.read(8))[0]
        
        tokens = None
        merges = None
        bos_id = eos_id = pad_id = -1
        
        for ki in range(n_kv):
            klen = struct.unpack('<Q', f.read(8))[0]
            key = f.read(klen).decode('utf-8')
            vtype = struct.unpack('<i', f.read(4))[0]
            
            if key == 'tokenizer.ggml.tokens' and vtype == 9:
                at = struct.unpack('<i', f.read(4))[0]
                al = struct.unpack('<Q', f.read(8))[0]
                if at == 8:
                    tokens = []
                    for _ in range(al):
                        sl = struct.unpack('<Q', f.read(8))[0]
                        tokens.append(f.read(sl))
                    # Write vocab as binary: [num_tokens:4] [len:4][bytes]...
                    outpath = os.path.join(out_dir, 'vocab.bin')
                    with open(outpath, 'wb') as out:
                        out.write(struct.pack('<I', len(tokens)))
                        for t in tokens:
                            out.write(struct.pack('<I', len(t)))
                            out.write(t)
                    print(f'Wrote {len(tokens)} tokens to {outpath}')
            elif key == 'tokenizer.ggml.merges' and vtype == 9:
                at = struct.unpack('<i', f.read(4))[0]
                al = struct.unpack('<Q', f.read(8))[0]
                if at == 8:
                    merges = []
                    for _ in range(al):
                        sl = struct.unpack('<Q', f.read(8))[0]
                        merges.append(f.read(sl))
                    # Write merges as binary: [num_merges:4] [len:4][string:len]...
                    outpath = os.path.join(out_dir, 'merges.bin')
                    with open(outpath, 'wb') as out:
                        out.write(struct.pack('<I', len(merges)))
                        for m in merges:
                            out.write(struct.pack('<I', len(m)))
                            out.write(m)
                    print(f'Wrote {len(merges)} merges to {outpath}')
            elif key == 'tokenizer.ggml.bos_token_id':
                val = struct.unpack('<i', f.read(4))[0]
                bos_id = val
            elif key == 'tokenizer.ggml.eos_token_id':
                val = struct.unpack('<i', f.read(4))[0]
                eos_id = val
            elif key == 'tokenizer.ggml.padding_token_id':
                val = struct.unpack('<i', f.read(4))[0]
                pad_id = val
            else:
                # Skip value
                if vtype == 8:
                    sl = struct.unpack('<Q', f.read(8))[0]
                    f.seek(sl, 1)
                elif vtype == 9:
                    at = struct.unpack('<i', f.read(4))[0]
                    al = struct.unpack('<Q', f.read(8))[0]
                    if at == 8:
                        for _ in range(al):
                            el = struct.unpack('<Q', f.read(8))[0]
                            f.seek(el, 1)
                    else:
                        elem_size = 1 if at in (0,1,7) else 2 if at in (2,3) else 8 if at in (10,11,12) else 4
                        f.seek(al * elem_size, 1)
                elif vtype in (4,5,6): f.seek(4, 1)
                elif vtype in (0,1,7): f.seek(1, 1)
                elif vtype in (10,11,12): f.seek(8, 1)
                elif vtype in (2,3): f.seek(2, 1)
                else: f.seek(4, 1)
        
        # Write special tokens
        outpath = os.path.join(out_dir, 'special_tokens.bin')
        with open(outpath, 'wb') as out:
            out.write(struct.pack('<iii', bos_id, eos_id, pad_id))
        print(f'Wrote special tokens BOS={bos_id} EOS={eos_id} PAD={pad_id}')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <gguf_path> <output_dir>')
        sys.exit(1)
    os.makedirs(sys.argv[2], exist_ok=True)
    extract_tokenizer(sys.argv[1], sys.argv[2])
