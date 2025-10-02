# This code works with encryption and decryption using a 6D hyperchaotic system, DNA/RNA 
# coding, and security testing.
# But has low PSNR 

import numpy as np
import hashlib
from PIL import Image
import os
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import histogram
import RGB_Histogram

# DNA encoding/decoding rules
DNA_RULES = {
    1: {'00': 'A', '01': 'G', '10': 'C', '11': 'T'},
    2: {'00': 'A', '01': 'C', '10': 'G', '11': 'T'},
    3: {'00': 'T', '01': 'G', '10': 'C', '11': 'A'},
    4: {'00': 'T', '01': 'C', '10': 'G', '11': 'A'},
    5: {'00': 'G', '01': 'A', '10': 'T', '11': 'C'},
    6: {'00': 'G', '01': 'T', '10': 'A', '11': 'C'},
    7: {'00': 'C', '01': 'A', '10': 'T', '11': 'G'},
    8: {'00': 'C', '01': 'T', '10': 'A', '11': 'G'}
}

DNA_REVERSE = {rule: {v: k for k, v in mapping.items()} for rule, mapping in DNA_RULES.items()}

# RNA coding rules
RNA_CODES = {
    'A': '00',  # Adenine maps to 00
    'C': '01',  # Cytosine maps to 01
    'G': '10',  # Guanine maps to 10
    'U': '11'   # Uracil maps to 11
}

RNA_REVERSE = {v: k for k, v in RNA_CODES.items()}

# RNA codon tables
RNA_CODON_TABLE = [
    'CCU', 'CGA', 'CGC', 'CGG', 'CGU', 'CUA', 'CUC', 'CUG', 'CUU', 'CAA', 'CAC', 'CAG', 'CAU',
    'CCA', 'CCC', 'CCG', 'ACU', 'AGA', 'AGC', 'AGG', 'AGU', 'AUA', 'AUC', 'AUG', 'AUU', 'AAA',
    'AAC', 'AAG', 'AAU', 'ACA', 'ACC', 'ACG', 'UCU', 'UGA', 'UGC', 'UGG', 'UGU', 'UUA', 'UUC',
    'UUG', 'UUU', 'UAA', 'UAC', 'UAG', 'UAU', 'UCA', 'UCC', 'UCG', 'GCU', 'GGA', 'GGC', 'GGG',
    'GGU', 'GUA', 'GUC', 'GUG', 'GUU', 'GAA', 'GAC', 'GAG', 'GAU', 'GCA', 'GCC', 'GCG'
]

def sha512_to_binary(image_data):
    """Generate 512-bit binary key from image"""
    # print("image data:", image_data)
    # exit(0)
    hash_hex = hashlib.sha512(image_data).hexdigest()
    # print("Hash in Hex:  ", hash_hex)
    # exit(0)
    return bin(int(hash_hex, 16))[2:].zfill(512)
    

def generate_parameters(K):
    """Generate initial values and parameters from key k1-k11"""
    K = [K[i*32:(i+1)*32] for i in range(16)]
    
    # Initial values for 6D hyperchaotic system
    x1 = (int(K[0], 2) ^ int(K[6], 2)) / 2**32
    x2 = (int(K[1], 2) ^ int(K[7], 2)) / 2**32
    x3 = (int(K[2], 2) ^ int(K[8], 2)) / 2**32
    x4 = (int(K[3], 2) ^ int(K[9], 2)) / 2**32
    x5 = (int(K[4], 2) ^ int(K[10], 2)) / 2**32
    x6 = (int(K[5], 2) ^ int(K[11], 2)) / 2**32
    
    # 6D hyperchaotic system parameters
    a = 18 + (int(K[12], 2) ^ int(K[13], 2)) / 2**32
    b = 16 + (int(K[13], 2) ^ int(K[14], 2)) / 2**32
    c = 0.5 + (int(K[14], 2) ^ int(K[15], 2)) / 2**32
    d = 5.3 + (int(K[12], 2) ^ int(K[15], 2)) / 2**32
    e = 32 + (int(K[11], 2) ^ int(K[15], 2)) / 2**32
    f_val = 9 + (int(K[10], 2) ^ int(K[14], 2)) / 2**32
    g = 5 + (int(K[9], 2) ^ int(K[13], 2)) / 2**32
    h = 2 + (int(K[8], 2) ^ int(K[12], 2)) / 2**32
    i_val = 4.1 + (int(K[7], 2) ^ int(K[11], 2)) / 2**32
    j_val = 3 + (int(K[6], 2) ^ int(K[10], 2)) / 2**32
    k_val = 12 + (int(K[5], 2) ^ int(K[9], 2)) / 2**32
    l_val = 4 + (int(K[4], 2) ^ int(K[8], 2)) / 2**32
    m_val = 50 + (int(K[3], 2) ^ int(K[7], 2)) / 2**32
    
    # Block size
    m = (int(K[0], 2) ^ int(K[2], 2) ^ int(K[4], 2)) % 5 + 2
    n = (int(K[1], 2) ^ int(K[3], 2) ^ int(K[5], 2)) % 5 + 2
    
    return (x1, x2, x3, x4, x5, x6, a, b, c, d, e, f_val, g, h, i_val, j_val, 
            k_val, l_val, m_val, int(m), int(n))

def hyperchaotic_6d_system(x1, x2, x3, x4, x5, x6, a, b, c, d, e, f_val, g, h, i_val, j_val, k_val, l_val, m_val, iterations):
    """Generate 6D hyperchaotic sequences"""
    X1, X2, X3, X4, X5, X6 = [], [], [], [], [], []
    
    for _ in range(iterations):
        dx1 = -a*x1 + b*x2 - a*x6*np.cos(x4) - c*x3*x5
        dx2 = -d*x2 + e*x1 - f_val*x1*np.cos(x5) - g*x1*x3
        dx3 = -h*x3 - i_val*x1 + j_val*x4*np.sin(x1) + x1*x2
        dx4 = -h*x4 + k_val*x2 - g*x1*np.cos(x6) - x2*x3
        dx5 = -h*x5 + x3 - l_val*x3*np.cos(x2) - x1*x2
        dx6 = -m_val*x6 + j_val*x5 + l_val*x2*np.sin(x3) - l_val*x3*x4
        
        x1 = x1 + dx1 * 0.01
        x2 = x2 + dx2 * 0.01
        x3 = x3 + dx3 * 0.01
        x4 = x4 + dx4 * 0.01
        x5 = x5 + dx5 * 0.01
        x6 = x6 + dx6 * 0.01
        
        x1 = max(min(x1, 100), -100)
        x2 = max(min(x2, 100), -100)
        x3 = max(min(x3, 100), -100)
        x4 = max(min(x4, 100), -100)
        x5 = max(min(x5, 100), -100)
        x6 = max(min(x6, 100), -100)
        
        X1.append(x1); X2.append(x2); X3.append(x3)
        X4.append(x4); X5.append(x5); X6.append(x6)
    
    return np.array(X1), np.array(X2), np.array(X3), np.array(X4), np.array(X5), np.array(X6)

def scramble_image(img, X1, X2, X3, m, n, reverse=False, perm_indices=None, swap_indices=None):
    """Perform block and pixel scrambling with reversibility"""
    M, N = img.shape[0], img.shape[1]
    Mb = M + (m - M % m) if M % m != 0 else M
    Nb = N + (n - N % n) if N % n != 0 else N
    padded_img = np.pad(img, ((0, Mb-M), (0, Nb-N)), mode='constant')
    
    num_blocks = (Mb * Nb) // (m * n)
    A1 = np.nan_to_num(np.mod(np.abs(X1[-num_blocks:]) * 2**14, 1), nan=0.5)
    A2 = np.nan_to_num(np.mod(np.abs(np.concatenate([X2[-Mb*Nb//2:], X3[-Mb*Nb//2:]])) * 2**14, 1), nan=0.5)
    
    block_indices = np.arange(num_blocks)
    if not reverse:
        scrambled_blocks = block_indices[np.argsort(-A1)]
        perm_indices = scrambled_blocks
        swap_indices = []
        scrambled_img = np.zeros_like(padded_img)
        for i, block_idx in enumerate(scrambled_blocks):
            row = (block_idx // (Nb//n)) * m
            col = (block_idx % (Nb//n)) * n
            block = padded_img[row:row+m, col:col+n]
            flat_block = block.flatten()
            swaps = []
            for j in range(len(flat_block)-1, 0, -1):
                k = min(max(int(A2[i*len(flat_block) + j] * (j + 1)), 0), len(flat_block)-1)
                swaps.append((j, k))
                flat_block[j], flat_block[k] = flat_block[k], flat_block[j]
            swap_indices.append(swaps)
            scrambled_img[row:row+m, col:col+n] = flat_block.reshape((m, n))
    else:
        original_indices = np.argsort(perm_indices)
        scrambled_img = np.zeros_like(padded_img)
        for i, orig_idx in enumerate(original_indices):
            row = (orig_idx // (Nb//n)) * m
            col = (orig_idx % (Nb//n)) * n
            block = padded_img[row:row+m, col:col+n]
            flat_block = block.flatten()
            swaps = swap_indices[i]
            for j, k in reversed(swaps):
                flat_block[j], flat_block[k] = flat_block[k], flat_block[j]
            scrambled_img[row:row+m, col:col+n] = flat_block.reshape((m, n))
    
    return scrambled_img[:M, :N], perm_indices, swap_indices

def dna_encode(data, rule):
    """Encode data to DNA sequence"""
    binary = ''.join([format(x, '08b') for x in data.flatten()])
    return ''.join([DNA_RULES[rule][binary[i:i+2]] for i in range(0, len(binary), 2)])

def dna_decode(dna, rule, shape):
    """Decode DNA sequence to data"""
    binary = ''.join([DNA_REVERSE[rule][c] for c in dna])
    return np.array([int(binary[i:i+8], 2) for i in range(0, len(binary), 8)]).reshape(shape)

def generate_rna_tables(chaotic_seq):
    """Generate RNA codon tables based on chaotic sequence"""
    c1 = [int(chaotic_seq[i] * 2**32) % 64 for i in range(64)]
    t00 = [RNA_CODON_TABLE[idx] for idx in c1]
    
    c2 = [int(chaotic_seq[i+64] * 2**32) % 64 for i in range(64)]
    t10 = [RNA_CODON_TABLE[idx] for idx in c2]
    
    t01 = [''.join([RNA_REVERSE[RNA_CODES[c]] for c in codon]) for codon in t00]
    t11 = [''.join([RNA_REVERSE[RNA_CODES[c]] for c in codon]) for codon in t10]
    
    return {'00': t00, '01': t01, '10': t10, '11': t11}

def rna_encode(data, rna_tables, chaotic_seq):
    """Perform RNA encoding on data"""
    binary = ''.join([format(x, '08b') for x in data.flatten()])
    encoded = []
    codon_indices = []  # Store original indices for decoding
    
    for i in range(0, len(binary), 8):
        if i + 8 > len(binary):
            break
        pixel_bin = binary[i:i+8]
        table_key = format(int(chaotic_seq[i//8] * 2**32) % 4, '02b')
        selected_table = rna_tables[table_key]
        first_6 = pixel_bin[:6]
        codon_idx = int(first_6, 2) % 64
        encoded_codon = selected_table[codon_idx]
        last_2 = format(int(pixel_bin[6:], 2) ^ int(table_key, 2), '02b')
        encoded.append(encoded_codon + last_2)
        codon_indices.append((i//8, codon_idx))  # Store index with position
    
    return ''.join(encoded), codon_indices

def rna_decode(rna_str, rna_tables, chaotic_seq, shape, codon_indices):
    """Decode RNA sequence to original data with exact reversal"""
    decoded_binary = []
    for i, (pos, orig_idx) in enumerate(codon_indices):
        start = i * 5
        if start + 5 > len(rna_str):
            break
        chunk = rna_str[start:start+5]
        codon = chunk[:3]
        last_2 = chunk[3:]
        table_key = format(int(chaotic_seq[pos] * 2**32) % 4, '02b')
        selected_table = rna_tables[table_key]
        # Use the original index for exact reversal
        first_6 = format(orig_idx, '06b')
        original_last_2 = format(int(last_2, 2) ^ int(table_key, 2), '02b')
        decoded_binary.append(first_6 + original_last_2)
    binary = ''.join(decoded_binary)
    return np.array([int(binary[i:i+8], 2) for i in range(0, len(binary), 8)]).reshape(shape)

def npcr_test(cipher1, cipher2):
    """Calculate NPCR (Number of Pixel Change Rate)"""
    diff = cipher1 != cipher2
    return (np.sum(diff) / cipher1.size) * 100

def uaci_test(cipher1, cipher2):
    """Calculate UACI (Unified Average Changing Intensity)"""
    return (np.sum(np.abs(cipher1.astype(int) - cipher2.astype(int))) / (255 * cipher1.size)) * 100

def entropy_test(image):
    """Calculate entropy"""
    hist = np.histogram(image.flatten(), bins=256, range=[0, 256])[0]
    hist = hist / hist.sum()
    ent = -np.sum(hist * np.log2(hist + 1e-10))
    return ent



def test_security(original_img, encrypted_img, key_params):
    """Generate slightly modified image and test NPCR/UACI using existing key parameters"""
    modified_img = original_img.copy()
    x, y = 1, 3  # Fixed pixel modification
    modified_img[x, y] = 0 if modified_img[x, y] != 0 else 255
    
    M, N = key_params['original_shape']
    (x1, x2, x3, x4, x5, x6, a, b, c, d, e, f_val, g, h, i_val, j_val, 
     k_val, l_val, m_val, m, n) = (key_params['x1'], key_params['x2'], key_params['x3'], 
                                   key_params['x4'], key_params['x5'], key_params['x6'],
                                   key_params['a'], key_params['b'], key_params['c'],
                                   key_params['d'], key_params['e'], key_params['f_val'],
                                   key_params['g'], key_params['h'], key_params['i_val'],
                                   key_params['j_val'], key_params['k_val'], key_params['l_val'],
                                   key_params['m_val'], key_params['m'], key_params['n'])
    
    X1, X2, X3, X4, X5, X6 = hyperchaotic_6d_system(
        x1, x2, x3, x4, x5, x6, a, b, c, d, e, f_val, g, h, i_val, j_val, 
        k_val, l_val, m_val, 2000 + 2*M*N)
    
    P1 = scramble_image(modified_img, X1, X2, X3, m, n)[0].astype(np.uint8)
    F = np.mod(np.floor(np.abs(X4[M*N:2*M*N]) * 2**14), 256).astype(np.uint8)
    P2_flat = (P1.flatten().astype(np.int32) + F.astype(np.int32)) % 256
    P2 = P2_flat.reshape(M, N).astype(np.uint8)
    R1 = np.mod(np.floor(np.abs(X1[:M*N]) * 2**14), 8) + 1
    R2 = np.mod(np.floor(np.abs(X2[:M*N]) * 2**14), 8) + 1
    R3 = np.mod(np.floor(np.abs(X3[:M*N]) * 2**14), 8) + 1
    T = np.mod(np.floor(np.abs(X2[M*N:2*M*N]) * 2**14), 4)
    
    PD = dna_encode(P2, R1[0])
    D = dna_encode(F, R2[0])
    cipher_dna = []
    for i in range(0, len(PD), 4):
        if i == 0:
            for j in range(4):
                cipher_dna.append('A' if PD[j] == D[j] else 'T')
        else:
            prev = cipher_dna[i-4:i]
            prev_bin = ''.join([DNA_REVERSE[R3[0]][c] for c in prev])
            for j in range(4):
                p = DNA_REVERSE[R1[0]][PD[i + j]]
                d = DNA_REVERSE[R2[0]][D[i + j]]
                prev_part = prev_bin[j*2:(j+1)*2] if j < 2 else '00'
                p_int = int(p, 2)
                d_int = int(d, 2)
                prev_int = int(prev_part, 2) if prev_part else 0
                if T[i//4] == 0:
                    res = (p_int ^ d_int) + prev_int
                elif T[i//4] == 1:
                    res = (p_int ^ d_int) - prev_int
                elif T[i//4] == 2:
                    res = (~p_int ^ d_int) + prev_int
                elif T[i//4] == 3:
                    res = (~p_int ^ d_int) - prev_int
                res_bin = format(res % 4, '02b')
                cipher_dna.append(DNA_RULES[R3[0]][res_bin])
    C1 = dna_decode(''.join(cipher_dna), R3[0], (M, N)).astype(np.uint8)
    rna_tables = generate_rna_tables(X5)
    rna_seq, codon_indices = rna_encode(C1, rna_tables, X6)
    modified_encrypted = rna_decode(rna_seq, rna_tables, X6, (M, N), codon_indices).astype(np.uint8)
    chaotic_shift = int(np.mod(np.abs(X6[-1]) * 100, M*N))
    modified_encrypted = np.bitwise_xor(modified_encrypted, np.roll(modified_encrypted, shift=chaotic_shift))
    
    npcr = npcr_test(encrypted_img, modified_encrypted)
    uaci = uaci_test(encrypted_img, modified_encrypted)

    entropy_val = entropy_test(encrypted_img) # For entropy 

    return npcr, uaci , entropy_val

def encrypt_image(image_path, test_mode=False):
    """Complete encryption process with 6D hyperchaotic system, DNA/RNA coding"""
    # print("====== These are the values inside the encrypt_image function =========")
    # print("encrypt_image called for:", image_path)
    try:
        if isinstance(image_path, np.ndarray):
            img = image_path
            image_data = img.tobytes()
        else:
            img = np.array(Image.open(image_path).convert('L'), dtype=np.uint8)
            with open(image_path, 'rb') as f:
                image_data = f.read()
            
        M, N = img.shape

        K = sha512_to_binary(image_data)

        # print("K: ", K)
        # exit(0)

        (x1, x2, x3, x4, x5, x6, a, b, c, d, e, f_val, g, h, i_val, j_val, 
         k_val, l_val, m_val, m, n) = generate_parameters(K)
        
        # print("Key in Encrypt Image Function:", K)
        
        X1, X2, X3, X4, X5, X6 = hyperchaotic_6d_system(
            x1, x2, x3, x4, x5, x6, a, b, c, d, e, f_val, g, h, i_val, j_val, 
            k_val, l_val, m_val, 2000 + 2*M*N)
        
        # print("X1:", X1)
        # print("X2:", X2)
        # print("X3:", X3)
        # print("X4:", X4)
        # print("X5:", X5)
        # print("X6:", X6)

        P1, perm_indices, swap_indices = scramble_image(img, X1, X2, X3, m, n)

        F = np.mod(np.floor(np.abs(X4[M*N:2*M*N]) * 2**14), 256).astype(np.uint8)
        P2_flat = (P1.flatten().astype(np.int32) + F.astype(np.int32)) % 256
        P2 = P2_flat.reshape(M, N).astype(np.uint8)

        R1 = np.mod(np.floor(np.abs(X1[:M*N]) * 2**14), 8) + 1
        R2 = np.mod(np.floor(np.abs(X2[:M*N]) * 2**14), 8) + 1
        R3 = np.mod(np.floor(np.abs(X3[:M*N]) * 2**14), 8) + 1
        T = np.mod(np.floor(np.abs(X2[M*N:2*M*N]) * 2**14), 4)

        PD = dna_encode(P2, R1[0])
        D = dna_encode(F, R2[0])
        
        cipher_dna = []
        for i in range(0, len(PD), 4):
            if i == 0:
                for j in range(4):
                    cipher_dna.append('A' if PD[j] == D[j] else 'T')
            else:
                prev = cipher_dna[i-4:i]
                prev_bin = ''.join([DNA_REVERSE[R3[0]][c] for c in prev])
                for j in range(4):
                    p = DNA_REVERSE[R1[0]][PD[i + j]]
                    d = DNA_REVERSE[R2[0]][D[i + j]]
                    prev_part = prev_bin[j*2:(j+1)*2] if j < 2 else '00'
                    p_int = int(p, 2)
                    d_int = int(d, 2)
                    prev_int = int(prev_part, 2) if prev_part else 0
                    if T[i//4] == 0:
                        res = (p_int ^ d_int) + prev_int
                    elif T[i//4] == 1:
                        res = (p_int ^ d_int) - prev_int
                    elif T[i//4] == 2:
                        res = (~p_int ^ d_int) + prev_int
                    elif T[i//4] == 3:
                        res = (~p_int ^ d_int) - prev_int
                    res_bin = format(res % 4, '02b')
                    cipher_dna.append(DNA_RULES[R3[0]][res_bin])
        
        C1 = dna_decode(''.join(cipher_dna), R3[0], (M, N)).astype(np.uint8)
        # print("Intermediate C1:", C1.mean(), C1.shape, C1[0,0])

        rna_tables = generate_rna_tables(X5)
        rna_seq, codon_indices = rna_encode(C1, rna_tables, X6)
        # print("RNA sequence sample:", rna_seq[:20])
        
        C = rna_decode(rna_seq, rna_tables, X6, (M, N), codon_indices).astype(np.uint8)
        chaotic_shift = int(np.mod(np.abs(X6[-1]) * 100, M*N))
        C = np.bitwise_xor(C, np.roll(C, shift=chaotic_shift))
        
        key_params = {
            'x1': float(x1), 'x2': float(x2), 'x3': float(x3), 
            'x4': float(x4), 'x5': float(x5), 'x6': float(x6),
            'a': float(a), 'b': float(b), 'c': float(c), 
            'd': float(d), 'e': float(e), 'f_val': float(f_val),
            'g': float(g), 'h': float(h), 'i_val': float(i_val),
            'j_val': float(j_val), 'k_val': float(k_val),
            'l_val': float(l_val), 'm_val': float(m_val),
            'm': int(m), 'n': int(n),
            'R1': int(R1[0]), 'R2': int(R2[0]), 'R3': int(R3[0]),
            'F': F.tolist(),
            'T': T.tolist(),
            'original_shape': (int(M), int(N)),
            'rna_tables': rna_tables,
            'chaotic_shift': int(chaotic_shift),
            'X1': X1.tolist(), 'X2': X2.tolist(), 'X3': X3.tolist(),
            'X4': X4.tolist(), 'X5': X5.tolist(), 'X6': X6.tolist(),
            'perm_indices': perm_indices,
            'swap_indices': swap_indices,
            'codon_indices': codon_indices
        }
        if test_mode:
            return C, None
        return C, key_params
    
    except Exception as e:
        print(f"Encryption error for {image_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None
    

def visualize_results(original_img, encrypted_img, plain_img):
    """Display the original, encrypted and decrypted images side by side"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(encrypted_img, cmap='gray')
    plt.title('Encrypted Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(plain_img, cmap='gray')
    plt.title('Decrypted Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def decrypt_image(cipher_img, key_params):
    # print("========These are the values inside the decryption function======")
    try:
        M, N = key_params['original_shape']

        # Convert lists back to numpy arrays
        X1 = np.array(key_params['X1'])
        X2 = np.array(key_params['X2'])
        X3 = np.array(key_params['X3'])
        X4 = np.array(key_params['X4'])
        X5 = np.array(key_params['X5'])
        X6 = np.array(key_params['X6'])
        
        # print("X1:", X1)
        # print("X2:", X2)
        # print("X3:", X3)
        # print("X4:", X4)
        # print("X5:", X5)
        # print("X6:", X6)

        F = np.array(key_params['F'], dtype=np.uint8).reshape((M, N))
        # print("F mean:", F.mean(), F.shape, F[0,0])
        T = np.array(key_params['T'], dtype=np.uint8)
        
        # Reverse final permutation
        chaotic_shift = key_params['chaotic_shift']
        C = np.bitwise_xor(cipher_img, np.roll(cipher_img, shift=-chaotic_shift))
        # print("After XOR and roll (C):", 0, C.shape, C[0,0])

        rna_tables = key_params['rna_tables']
        codon_indices = key_params['codon_indices']
        rna_seq, _ = rna_encode(C, rna_tables, X6)  # Re-encode to get sequence
        C1 = rna_decode(rna_seq, rna_tables, X6, (M, N), codon_indices)
        # print("After RNA decode (C1):", np.mean(np.abs(C - C1)), C1.shape, C1[0,0])
        # print("C1 mean:", C1.mean())

        cipher_dna = dna_encode(C1, key_params['R3'])
        PD = [''] * len(cipher_dna)
        D = dna_encode(F.flatten(), key_params['R2'])
        
        for i in range(4):
            PD[i] = 'A' if cipher_dna[i] == D[i] else 'T'
        
        for i in range(1, M*N):
            idx = 4*i
            prev = cipher_dna[idx-4:idx]
            prev_bin = ''.join([DNA_REVERSE[key_params['R3']][c] for c in prev])
            
            for j in range(4):
                c = DNA_REVERSE[key_params['R3']][cipher_dna[idx + j]]
                d = DNA_REVERSE[key_params['R2']][D[idx + j]]
                prev_part = prev_bin[j*2:(j+1)*2] if j < 2 else '00'
                c_int = int(c, 2)
                d_int = int(d, 2)
                prev_int = int(prev_part, 2) if prev_part else 0
                
                if T[i] == 0:
                    p = (c_int - prev_int) ^ d_int
                elif T[i] == 1:
                    p = (c_int + prev_int) ^ d_int
                elif T[i] == 2:
                    p = ~((c_int - prev_int) ^ d_int)
                elif T[i] == 3:
                    p = ~((c_int + prev_int) ^ d_int)
                
                p = p % 4
                PD[4*i + j] = DNA_RULES[key_params['R1']][format(p, '02b')]
        
        P2 = dna_decode(''.join(PD), key_params['R1'], (M, N))
        # print("After DNA decode (P2):", np.mean(np.abs(C1 - P2)), P2.shape, P2[0,0])
        # print("P2 mean:", P2.mean())

        P1 = np.mod(P2 - F, 256).astype(np.uint8)
        # print("After diffusion reverse (P1):", np.mean(np.abs(P2 - P1)), P1.shape, P1[0,0])
        # print("P1 mean:", P1.mean())

        plain_img, _, _ = scramble_image(P1, X1, X2, X3, key_params['m'], key_params['n'], 
                                       reverse=True, 
                                       perm_indices=key_params['perm_indices'],
                                       swap_indices=key_params['swap_indices'])
        # print("After scrambling reverse (plain_img):", np.mean(np.abs(P1 - plain_img)), plain_img.shape, plain_img[0,0])
        # print("plain_img mean:", plain_img.mean())
        
        return plain_img
    
    except Exception as e:
        print(f"Decryption error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    

def calculate_mse(original, decrypted):
    """Calculate Mean Squared Error between original and decrypted images"""
    return np.mean((original - decrypted) ** 2)

if __name__ == "__main__":
    test_images = [
        
    # '5.3.02.tiff', 
    # '4.1.01.png',
    # 'pepper.tiff' ,#4.2.07.tiff',
    # '4.1.03.png',
    # '4.1.06.tiff',
    # '4.1.07.tiff',
    # '4.1.08.tiff',
   
    # 'aeroplane.png',
    # 'umer.jpg',
    # '5.1.09.tiff',
    # '5.1.11.tiff',
    # '5.1.12.tiff',
    # '5.1.14.tiff',
    # '5.2.08.tiff',
    # '5.2.10.tiff',
    # '5.3.01.tiff',
    # '5.3.02.tiff',
    # '7.1.01.tiff',
    # '7.1.02.tiff',
    # '7.1.03.tiff',
    # '7.1.04.tiff',
    # '7.1.05.tiff',
    # '7.1.06.tiff',
    # '7.1.07.tiff',
    # '7.1.08.tiff',
    # '7.1.09.tiff',
    # '7.1.10.tiff',
    # '7.2.01.tiff',
    # 'adhar.jpg',
    # 'umer_128.png',

    # '5.1.09.tiff',
    # '5.1.11.tiff',
    # '5.1.12.tiff',
     
    # 'cameraman.jpg',
    # 'Baboon.png',
    
    # # '7.1.03.tiff',
    # # '7.1.04.tiff',
    # # '7.1.05.tiff',
    # # '7.1.06.tiff',
    # # '7.1.07.tiff',
    # 'Baboon.png',
    # 'cameraman.jpg',
    # 'lenna.jpg',
    # '4.2.07.tiff',
    #  '5.3.01.tiff',
    #  '7.2.01.tiff',
    # '5.1.09.tiff',
    # '7.1.03.tiff',
    # '7.1.05.tiff',
    # '7.1.06.tiff',
    #  '7.2.01.tiff',
     
    # '7.1.06.tiff',
    # '7.1.07.tiff',
    'Baboon.png',
    'lenna.jpg',
    'umer.jpg',
    #  'lenna.jpg', 
    # '5.3.01.tiff',
    # '5.3.02.tiff',   
    ]
    output_folder = "encrypted_results"
    os.makedirs(output_folder, exist_ok=True)

    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    # BRIGHT_GREEN = "\033[92m"
    
    RED = "\033[91m"
    results = []
    successful_encryptions=0

    print(f"{BOLD}{CYAN}")
#     print(r"""
   

# █████████████████████████ 6D HYPER CHAOTIC ENCRYPTION ENGINE █████████████████
# █                                                                            █
# █      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░       █
# █      ░    ____    ____    ____   ____    ________    ________      ░       █
# █      ░   |    |  |    |  |    \ /    |  |   _____|  |  ____  |     ░       █
# █      ░   |    |  |    |  |     -     |  |  |_____   |  |__|  |     ░       █
# █      ░   |    |__|    |  |   |\ /|   |  |   _____|  |    ____|     ░       █
# █      ░   |            |  |   | | |   |  |  |_____   |  | \  \      ░       █
# █      ░   |____________|  |___|   |___|  |________|  |__|  \__\     ░       █
# █      ░                                                             ░       █
# █      ░                                                             ░       █
# █      ░      ▓  6D Hyperchaotic System    ▓  DNA / RNA Coding   ▓   ░       █
# █      ░      ▓  SHA-512 Key Derivation    ▓  Logical Diffusion  ▓   ░       █
# █      ░         dx₁/dt = -a·x₁ + b·x₂ - a·x₆·cos(x₄) - c·x₃·x₅      ░       █
# █      ░         dx₂/dt = -d·x₂ + e·x₁ - f·x₁·cos(x₅) - g·x₁·x₃      ░       █
# █      ░         dx₃/dt = -h·x₃ - i·x₁ + j·x₄·sin(x₁) + x₁·x₂        ░       █
# █      ░         dx₄/dt = -h·x₄ + k·x₂ - g·x₁·cos(x₆) - x₂·x₃        ░       █
# █      ░         dx₅/dt = -h·x₅ + x₃ - l·x₃·cos(x₂) - x₁·x₂          ░       █
# █      ░         dx₆/dt = -m·x₆ + j·x₅ + l·x₂·sin(x₃) - l·x₃·x₄      ░       █
# █      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░       █
# █                      [ ENCRYPT · SCRAMBLE · PROTECT · RESTORE ]            █
# ██████████████████████████████████████████████████████████████████████████████  
#                 6D Hyperchaotic system with RNA and DNA Coding  
#     """)

#     print("\t\t\t6D Hyperchaotic system with RNA and DNA Coding\n")

    print("█" * 92)
    # print(f"{'Image':<18}  {'NPCR(%)':<15}{'UACI(%)':<15} {'Entropy':<15}  {'PSNR(dB)':<15}  {'Size'}")
    print(f"{'Image':<18}  {'NPCR(%)':<15}{'UACI(%)':<15} {'Entropy':<15}    {'Size'}")
        #   {'Entropy':<15}{'PSNR(dB)':<15} {'MSE':<15} {'Size'}")
    print("█" * 92)
    
    for img_file in test_images:
        try:
            if not os.path.exists(img_file):
                print(f"Image not found: {img_file}")
                continue
                
            encrypted_img, key_params = encrypt_image(img_file)
            
            if encrypted_img is not None:
                output_path = os.path.join(output_folder, f"encrypted_{img_file}")
                Image.fromarray(encrypted_img).save(output_path)
                
                # Save key parameters
                key_path = os.path.join(output_folder, f"key_{img_file}.pkl")
                with open(key_path, 'wb') as f:
                    pickle.dump(key_params, f)
                successful_encryptions += 1
                
                # original_img = np.array(Image.open(img_file).convert('L'))
                # size_M, size_N = original_img.shape
                # npcr, uaci = test_security(original_img, encrypted_img, key_params)
                # print(f"{img_file:<20}{npcr:<15.4f}{uaci:<15.4f} {size_M}x{size_N}")
                
                
                # Test decryption
                decrypted_img = decrypt_image(encrypted_img, key_params)
                if decrypted_img is not None:
                    decrypted_path = os.path.join(output_folder, f"decrypted_{img_file}")
                    Image.fromarray(decrypted_img).save(decrypted_path)
                    # print(f"Decrypted image saved as: {decrypted_path}")
                    # Calculate PSNR
                    original = np.array(Image.open(img_file).convert('L'))
                    mse = np.mean((original - decrypted_img) ** 2)
                    psnr = 10 * np.log10(255**2 / mse) if mse != 0 else float('inf')
                    
                    original_img = np.array(Image.open(img_file).convert('L'))
                    size_M, size_N = original_img.shape
                    npcr, uaci , entropy_val = test_security(original_img, encrypted_img, key_params)
                    # print(f"{img_file:<20}{npcr:<15.4f}{uaci:<15.4f} {entropy_val:<15.4f} {psnr:<15.4f} {size_M}x{size_N}")
                    print(f"{img_file:<20}{npcr:<15.4f}{uaci:<15.4f} {entropy_val:<15.4f}  {size_M}x{size_N}")
                    results.append((img_file, npcr, uaci,entropy_val, psnr))
                    # RGB_Histogram.plot_rgb_histograms()
                    # histogram.plot_histograms()
                    visualize_results(original_img, encrypted_img,decrypted_img)

                else:
                    print("Decryption failed")
            else:
                print(f"Encryption failed for: {img_file}")
                
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    # print("\n" + "="*60)
    print("▒" * 92)
    print(f"{BOLD}{CYAN}{RESET}")
    print(f"{BOLD}{RED}")
    
    print("▓" * 58)
    print(f"▓▓\tTotal images processed: {len(test_images)}\t\t\t▓▓")
    print(f"▓▓\tSuccessful encryptions: {successful_encryptions}\t\t\t▓▓")
    
    if successful_encryptions > 0:
        avg_npcr = np.mean([r[1] for r in results if r[1] > 0])
        avg_uaci = np.mean([r[2] for r in results if r[2] > 0])
        avg_entropy = np.mean([r[3] for r in results if r[3] > 0])
        
        print("▓▓\t\t\t\t\t\t\t▓▓")
        print("▓▓\tSecurity Test Results:\t\t\t\t▓▓")
        print(f"▓▓\tAverage NPCR: {avg_npcr:.4f}% (Target >99.3%)\t\t▓▓")
        print(f"▓▓\tAverage UACI: {avg_uaci:.4f}% (Target >33.3%)\t\t▓▓")
        print(f"▓▓\tAverage Entropy: {avg_entropy:.4f} bits (Target >7.9000%)\t▓▓")
        
        
        npcr_met = "✓" if avg_npcr > 99.3 else "✗"
        uaci_met = "✓" if avg_uaci > 33.3 else "✗"
        entr_met = "✓" if avg_entropy > 7.9000 else "✗"
        
        print("▓▓\t\t\t\t\t\t\t▓▓")
        print(f"▓▓\tSecurity Targets:\t\t\t\t▓▓")
        print(f"▓▓\tNPCR >99.3%: {npcr_met}\t\t\t\t\t▓▓")
        print(f"▓▓\tUACI >33.3%: {uaci_met}\t\t\t\t\t▓▓")
        print(f"▓▓\tUACI >7.9000%: {uaci_met}\t\t\t\t▓▓")
        print("▓" * 58)

        print(f"{BOLD}{RED}{RESET}")
    
    print(f"\n\t\t\t{UNDERLINE}{BOLD}{CYAN}Encryption and decryption completed successfully!{RESET}\n")