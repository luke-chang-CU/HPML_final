import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn.functional as F
from models.model import GPT, GPTConfig
from models.vqvae import VQVAE
from data.data_utils import load_palette
import time
from PIL import Image
import numpy as np
import argparse

def rollback_kv(past_kvs, num_to_keep):
    if past_kvs is None:
        return None
    new_kvs = []
    for (k, v) in past_kvs:
        k_sliced = k[..., :num_to_keep, :]
        v_sliced = v[..., :num_to_keep, :]
        new_kvs.append((k_sliced, v_sliced))
    return new_kvs

@torch.no_grad()
def speculative_sampling(teacher, student, idx, max_new_tokens, gamma=4, temperature=1.0):
    # doing speculative decoding with the student drafting and teacher verifying.
    # unconditional mode so no class labels passed around.
    _, _, teacher_kv = teacher(idx, use_cache=True)
    _, _, student_kv = student(idx, use_cache=True)
    
    curr_idx = idx
    block_size = teacher.config.block_size
    
    generated_count = 0
    total_drafted = 0
    total_accepted = 0
    
    print(f"Starting generation loop. Goal: {max_new_tokens} tokens.")
    
    while generated_count < max_new_tokens:
        ctx_len = curr_idx.size(1)

        # --- Drafting (Student) ---
        draft_tokens = []
        draft_probs = []
        curr_student_kv = student_kv 
        curr_draft_seq = curr_idx
        
        for k in range(gamma):
            if curr_draft_seq.size(1) >= block_size:
                break
            last_token = curr_draft_seq[:, -1:]
            
            # Forward Student
            logits_S, _, new_student_kv = student(last_token, past_kvs=curr_student_kv, use_cache=True)
            logits_S = logits_S[:, -1, :] / temperature
            probs_S = F.softmax(logits_S, dim=-1)
            next_token = torch.multinomial(probs_S, num_samples=1)
            
            draft_tokens.append(next_token)
            draft_probs.append(probs_S)
            
            curr_draft_seq = torch.cat((curr_draft_seq, next_token), dim=1)
            curr_student_kv = new_student_kv
            
        if len(draft_tokens) == 0:
            break
            
        total_drafted += len(draft_tokens)

        # --- Verification Step (Teacher) ---
        draft_seq_tensor = torch.cat(draft_tokens, dim=1)
        # Teacher checks all the drafts at once which is why this is faster (parallel)
        logits_T_all, _, new_teacher_kv = teacher(draft_seq_tensor, past_kvs=teacher_kv, use_cache=True, all_logits=True)
        
        accepted_count = 0
        
        for i in range(len(draft_tokens)):
            token = draft_tokens[i]
            
            logits_T = logits_T_all[:, i, :] / temperature
            probs_T = F.softmax(logits_T, dim=-1)
            probs_S = draft_probs[i]
            
            p_T_val = probs_T.gather(1, token)
            p_S_val = probs_S.gather(1, token)
            
            # Rejection Sampling
            # If p_S <= p_T, accept. Else accept with ratio.
            # Avoid div by zero
            ratio = p_T_val / (p_S_val + 1e-10)
            
            # print(f"  Token {token.item()}: P_S={p_S_val.item():.4f}, P_T={p_T_val.item():.4f}, Ratio={ratio.item():.4f} -> ", end="")
            
            if p_S_val <= p_T_val:
                # print("ACCEPT (Auto)")
                accept = True
            else:
                r = torch.rand_like(p_T_val)
                accept = (r < ratio).item()
                # if accept:
                #     print(f"ACCEPT (Prob r={r.item():.4f})")
                # else:
                #     print(f"REJECT (Prob r={r.item():.4f})")

            if accept:
                curr_idx = torch.cat((curr_idx, token), dim=1)
                accepted_count += 1
                generated_count += 1
                if generated_count >= max_new_tokens:
                    teacher_kv = rollback_kv(new_teacher_kv, ctx_len + accepted_count)
                    return curr_idx, total_accepted / total_drafted
            else:
                # Rejected!
                # Sad times. We resample from the difference distribution.
                # basically corrects the student's mistake.
                diff = F.relu(probs_T - probs_S)
                if diff.sum() > 0:
                    diff = diff / diff.sum(dim=-1, keepdim=True)
                    new_token = torch.multinomial(diff, num_samples=1)
                else:
                    new_token = torch.multinomial(probs_T, num_samples=1)
                
                curr_idx = torch.cat((curr_idx, new_token), dim=1)
                generated_count += 1
                
                # Rollback Teacher to right before this correction
                valid_len = ctx_len + accepted_count
                teacher_kv = rollback_kv(new_teacher_kv, valid_len)
                
                # Forward correction token
                if generated_count < max_new_tokens:
                     _, _, teacher_kv = teacher(new_token, past_kvs=teacher_kv, use_cache=True)
                
                # Sync Student
                student_kv = rollback_kv(student_kv, ctx_len)
                segment = torch.cat((draft_seq_tensor[:, :i], new_token), dim=1)
                if segment.size(1) > 0:
                     _, _, student_kv = student(segment, past_kvs=student_kv, use_cache=True)
                
                break 
        else:
            # All Accepted
            last_logit_T = logits_T_all[:, -1, :] / temperature
            last_prob_T = F.softmax(last_logit_T, dim=-1)
            extra_token = torch.multinomial(last_prob_T, num_samples=1)
            
            curr_idx = torch.cat((curr_idx, extra_token), dim=1)
            generated_count += 1
            
            teacher_kv = new_teacher_kv
            if generated_count < max_new_tokens:
                 _, _, teacher_kv = teacher(extra_token, past_kvs=teacher_kv, use_cache=True)
            
            student_kv = curr_student_kv
            if generated_count < max_new_tokens:
                 _, _, student_kv = student(extra_token, past_kvs=student_kv, use_cache=True)
        
        total_accepted += accepted_count

    if total_drafted > 0:
        acceptance_rate = total_accepted / total_drafted
    else: 
        acceptance_rate = 0.0
        
    return curr_idx, acceptance_rate

def run_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load Valid Start Tokens
    token_path = 'dataset_tokens/dogs_08_17_tokens.pt'
    data = torch.load(token_path)
    if isinstance(data, dict):
        all_tokens = data['tokens']
    else:
        all_tokens = data
    
    # Pick random valid start token from dataset
    idx = torch.randint(0, len(all_tokens), (1,))
    start_token = all_tokens[idx, 0:1].to(device)
    print(f"Initialized with token: {start_token.item()}")

    # 2. Load Teacher (Unconditional 20L)
    print("Loading Teacher (Epoch 100)...")
    t_conf = GPTConfig(1024, 256, 20, 16, 1024, num_classes=None)
    teacher = GPT(t_conf)
    t_ckpt = torch.load('checkpoints_teacher_unconditional/teacher_epoch_100.pt', map_location=device)
    t_new = {k[10:] if k.startswith('_orig_mod.') else k: v for k,v in t_ckpt.items()}
    teacher.load_state_dict(t_new)
    teacher.to(device)
    teacher.eval()
    
    # 3. Load Student (Unconditional 10L)
    print("Loading Student...") 
    s_conf = GPTConfig(1024, 256, 10, 16, 1024, num_classes=None)
    student = GPT(s_conf)
    # Use Epoch 100
    s_path = 'checkpoints_distill_hybrid/student_10L_1024E_epoch_100.pt' 
    s_ckpt = torch.load(s_path, map_location=device)
    s_new = {k[10:] if k.startswith('_orig_mod.') else k: v for k,v in s_ckpt.items()}
    student.load_state_dict(s_new)
    student.to(device)
    student.eval()
    
    # 4. Run Speculative Decoding
    print("-" * 50)
    print("Running Speculative Decoding (Gamma=4, T=1.0)...")
    start = time.time()
    out, rate = speculative_sampling(teacher, student, start_token, max_new_tokens=255, gamma=4, temperature=1.0)
    end = time.time()
    
    duration = end - start
    speed = 255 / duration
    print(f"Done in {duration:.2f}s")
    print(f"Speed: {speed:.2f} tokens/s")
    print(f"Acceptance Rate: {rate*100:.2f}%")
    print("-" * 50)

    # 5. Decode Image (Optional visual check)
    vqvae = VQVAE(num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=256,
                  num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
    vqvae.load_state_dict(torch.load('checkpoints_vqvae_miniimagenet/vqvae_final.pt', map_location='cpu'))
    vqvae.to('cpu')
    vqvae.eval()
    
    indices = out.cpu().view(1, 256)
    decoded = vqvae.decode(indices)
    img = decoded[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    Image.fromarray((img * 255).astype(np.uint8)).save('speculative_unconditional_result.png')
    print("Saved result to 'speculative_unconditional_result.png'")

if __name__ == "__main__":
    run_benchmark()
