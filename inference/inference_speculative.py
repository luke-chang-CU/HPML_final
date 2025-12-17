import torch
import torch.nn.functional as F
from model import GPT, GPTConfig
from vqvae import VQVAE
from data_utils import tokens_to_image
import time
import os
from PIL import Image
import numpy as np

def rollback_kv(past_kvs, num_to_keep):
    """
    Slices the KV cache to keep only the first 'num_to_keep' tokens.
    Structure: List[Tuple[Tensor, Tensor]]
    Tensor: (B, nh, T, hs)
    """
    if past_kvs is None:
        return None
    new_kvs = []
    for (k, v) in past_kvs:
        k_sliced = k[..., :num_to_keep, :]
        v_sliced = v[..., :num_to_keep, :]
        new_kvs.append((k_sliced, v_sliced))
    return new_kvs

@torch.no_grad()
def speculative_sampling(teacher, student, idx, max_new_tokens, gamma=4, temperature=1.0, class_labels=None):
    """
    Speculative sampling with KV Caching for both models.
    """
    # 1. Initialize KV Caches with the prefix (everything in idx so far)
    # Forward the full prefix to get initial state
    # teacher returns logits for last pos, but we mainly want the cache
    _, _, teacher_kv = teacher(idx, class_labels=class_labels, use_cache=True)
    _, _, student_kv = student(idx, class_labels=class_labels, use_cache=True)
    
    # We are generating tokens one by one (or chunks)
    # Effectively, 'idx' grows. We need to keep track of current context length.
    
    curr_idx = idx
    block_size = teacher.config.block_size
    
    generated_count = 0
    total_drafted = 0
    total_accepted = 0
    
    while generated_count < max_new_tokens:
        ctx_len = curr_idx.size(1)

        # 2. Draft Tokens (Student)
        # We need to draft 'gamma' tokens.
        # This part is serial (auto-regressive) for the Student.
        
        draft_tokens = []
        draft_probs = []
        
        # We need a working copy of student_kv for the draft loop
        # Because if we reject, we discard this speculative extension
        # BUT, since 'model.py' returns NEW list with NEW tensors (cat result).
        # So 'curr_student_kv' can diverge without affecting 'student_kv' base?
        # Yes, as long as we don't mutate tensors in place (we don't, we prefer cat).
        
        curr_student_kv = student_kv 
        curr_draft_seq = curr_idx
        
        for k in range(gamma):
            # Check for context overflow
            if curr_draft_seq.size(1) >= block_size:
                break
                
            # Forward Student on LAST token only
            last_token = curr_draft_seq[:, -1:]
            
            logits_S, _, new_student_kv = student(last_token, class_labels=class_labels, past_kvs=curr_student_kv, use_cache=True)
            
            logits_S = logits_S[:, -1, :] / temperature
            probs_S = F.softmax(logits_S, dim=-1) # (B, V)
            
            next_token = torch.multinomial(probs_S, num_samples=1) # (B, 1)
            
            draft_tokens.append(next_token)
            draft_probs.append(probs_S) # Store full distribution for rejection sampling
            
            curr_draft_seq = torch.cat((curr_draft_seq, next_token), dim=1)
            curr_student_kv = new_student_kv
            
        # If we drafted 0 tokens (due to context limit), break or finish
        if len(draft_tokens) == 0:
            break
            
        # 3. Verification (Teacher)
        # Parallel forward on the draft sequence
        # We pass ALL draft tokens at once to the teacher
        # Teacher context: 'idx' (prefix) already cached in 'teacher_kv'
        # Input to teacher: the sequence of draft tokens
        
        draft_seq_tensor = torch.cat(draft_tokens, dim=1) # (B, gamma)
        
        # We need logits for ALL these positions
        # teacher(draft_seq) with past=teacher_kv
        # returns logits for draft_seq (B, gamma, V) if all_logits=True
        
        logits_T_all, _, new_teacher_kv = teacher(draft_seq_tensor, class_labels=class_labels, past_kvs=teacher_kv, use_cache=True, all_logits=True)
        # logits_T_all shape: (B, gamma, V)
        
        accepted_count = 0
        all_accepted = True
        
        # Verify loop
        for i in range(len(draft_tokens)):
            token = draft_tokens[i] # (B, 1) usually from multinomial
            # We need p_T(x | prefix + drafts[:i])
            # This corresponds to logits_T_all[:, i, :] 
            
            logits_T = logits_T_all[:, i, :] / temperature
            probs_T = F.softmax(logits_T, dim=-1) # (B, V)
            probs_S = draft_probs[i]           # (B, V)
            
            p_T_val = probs_T.gather(1, token) # p_T(token)
            p_S_val = probs_S.gather(1, token) # p_S(token)
            
            # Rejection Sampling
            r = torch.rand_like(p_T_val)
            ratio = p_T_val / (p_S_val + 1e-10)
            
            if r < torch.min(torch.ones_like(ratio), ratio):
                # Accepted
                curr_idx = torch.cat((curr_idx, token), dim=1)
                accepted_count += 1
                generated_count += 1
                if generated_count >= max_new_tokens:
                    # Limit reached immediately
                    teacher_kv = rollback_kv(new_teacher_kv, ctx_len + accepted_count) # Keep up to current
                    return curr_idx
            else:
                # Rejected
                all_accepted = False
                
                # Sample from residual distribution
                diff = F.relu(probs_T - probs_S)
                if diff.sum() > 0:
                    diff = diff / diff.sum(dim=-1, keepdim=True)
                    new_token = torch.multinomial(diff, num_samples=1)
                else:
                    new_token = torch.multinomial(probs_T, num_samples=1)
                
                curr_idx = torch.cat((curr_idx, new_token), dim=1)
                generated_count += 1
                
                # Update KV for Rejected case
                valid_len = ctx_len + accepted_count
                teacher_kv = rollback_kv(new_teacher_kv, valid_len)
                
                # Forward Teacher on Correction Token
                if generated_count < max_new_tokens:
                     _, _, teacher_kv = teacher(new_token, class_labels=class_labels, past_kvs=teacher_kv, use_cache=True)
                
                # Update Student KV: rollback to ctx_len (start of round)
                student_kv = rollback_kv(student_kv, ctx_len)
                
                # Forward correction token sequence through student to sync
                # We need to forward [accepted_drafts + correction]
                segment = torch.cat((draft_seq_tensor[:, :i], new_token), dim=1)
                if segment.size(1) > 0:
                     _, _, student_kv = student(segment, class_labels=class_labels, past_kvs=student_kv, use_cache=True)
                
                break 
        else:
            # All gamma tokens accepted
            # Sample one extra token from the LAST teacher distribution
            last_logit_T = logits_T_all[:, -1, :] / temperature
            last_prob_T = F.softmax(last_logit_T, dim=-1)
            extra_token = torch.multinomial(last_prob_T, num_samples=1)
            
            curr_idx = torch.cat((curr_idx, extra_token), dim=1)
            generated_count += 1
            
            # Update KVs
            teacher_kv = new_teacher_kv
            if generated_count < max_new_tokens:
                 _, _, teacher_kv = teacher(extra_token, class_labels=class_labels, past_kvs=teacher_kv, use_cache=True)
            
            # Sync Student KV
            student_kv = curr_student_kv
            if generated_count < max_new_tokens:
                 _, _, student_kv = student(extra_token, class_labels=class_labels, past_kvs=student_kv, use_cache=True)
        
        total_accepted += accepted_count
                 
    if total_drafted > 0:
        acceptance_rate = total_accepted / total_drafted
        print(f"Acceptance Rate: {acceptance_rate:.2f} ({total_accepted}/{total_drafted})")
    return curr_idx

import argparse

def run_speculative_decoding():
    parser = argparse.ArgumentParser()
    parser.add_argument('--student_id', type=int, default=1, help='1=12L, 2=8L, 3=6L, 4=4L')
    parser.add_argument('--gamma', type=int, default=4, help='Number of draft tokens')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of images to generate')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load Teacher
    print("Loading Teacher...")
    teacher_config = GPTConfig(1024, 256, 20, 16, 1024, num_classes=64)
    teacher = GPT(teacher_config)
    
    # Fix state dict loading for compiled model (Teacher)
    ckpt_T = torch.load('checkpoints_teacher_vqvae/teacher_final.pt', map_location=device)
    new_ckpt_T = {}
    for k, v in ckpt_T.items():
        if k.startswith('_orig_mod.'):
            new_ckpt_T[k[10:]] = v
        else:
            new_ckpt_T[k] = v
    teacher.load_state_dict(new_ckpt_T)
    teacher.to(device)
    teacher.eval()
    
    # Student Configs
    student_configs = {
        1: ("12L_768E", GPTConfig(1024, 256, 12, 12, 768, num_classes=64)),
        2: ("8L_512E", GPTConfig(1024, 256, 8, 8, 512, num_classes=64)),
        3: ("6L_384E", GPTConfig(1024, 256, 6, 6, 384, num_classes=64)),
        4: ("4L_256E", GPTConfig(1024, 256, 4, 4, 256, num_classes=64)),
    }
    
    name, config = student_configs[args.student_id]
    print(f"Loading Student: {name}")
    student = GPT(config)
    
    ckpt_path_S = f'checkpoints_distill_vqvae/student_{name}_final.pt'
    if not os.path.exists(ckpt_path_S):
        print(f"Student checkpoint {ckpt_path_S} not found.")
        return

    # Fix state dict loading for compiled model (Student)
    ckpt_S = torch.load(ckpt_path_S, map_location=device)
    new_ckpt_S = {}
    for k, v in ckpt_S.items():
        if k.startswith('_orig_mod.'):
            new_ckpt_S[k[10:]] = v
        else:
            new_ckpt_S[k] = v
    student.load_state_dict(new_ckpt_S)
    student.to(device)
    student.eval()
    
    # Load VQ-VAE for decoding
    vqvae = VQVAE(num_hiddens=1024, num_residual_layers=2, num_residual_hiddens=256,
                  num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
    if os.path.exists('checkpoints_vqvae_miniimagenet/vqvae_final.pt'):
        vqvae.load_state_dict(torch.load('checkpoints_vqvae_miniimagenet/vqvae_final.pt', map_location='cpu'))
    vqvae.to('cpu')
    vqvae.eval()
    
    print(f"Starting speculative decoding with Gamma={args.gamma}...")
    
    for i in range(args.num_samples):
        initial_idx = torch.randint(0, 1024, (1, 1)).to(device)
        random_label = torch.randint(0, 64, (1,), device=device)
        print(f"Generating sample {i+1}/{args.num_samples} (Class {random_label.item()})...")
        
        start_time = time.time()
        output_tokens = speculative_sampling(teacher, student, initial_idx, max_new_tokens=255, gamma=args.gamma, temperature=1.0, class_labels=random_label)
        end_time = time.time()
        
        total_time = end_time - start_time
        num_tokens = output_tokens.shape[1] - 1
        print(f"Generated {num_tokens} tokens in {total_time:.4f}s ({num_tokens/total_time:.2f} tokens/s)")
        
        # Decode image
        indices = output_tokens[0].cpu().view(1, -1)
        # Ensure 256 length
        if indices.shape[1] > 256:
             indices = indices[:, :256]
        
        decoded = vqvae.decode(indices)
        img_tensor = decoded[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        img = (img_tensor * 255).astype(np.uint8)
        
        filename = f'speculative_sample_{name}_{i}.png'
        Image.fromarray(img).save(filename)
        print(f"Saved {filename}")

if __name__ == "__main__":
    run_speculative_decoding()
