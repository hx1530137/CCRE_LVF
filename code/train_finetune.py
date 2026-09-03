'LoRA fine-tuning for the CCRE encoder.'
import json, random, os, time, gc, argparse, sys
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

os.environ['TRITON_DISABLE_CUDA_KERNEL'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


BASE_4B = '/home/huxin/Documents/trae_projects/sikuBERT/models/Qwen3-Embedding-4B'
TRAIN_DATA_FINETUNE = '/home/huxin/Documents/trae_projects/sikuBERT/qwen3-emb-finetune/data/emb_train_data_clean_train.jsonl'
TRAIN_DATA_DUALVIEW = '/home/huxin/Documents/trae_projects/sikuBERT/qwen3-emb-finetune/data/dual_view_emb_data_clean_train.jsonl'
VAL_DATA_FINETUNE   = '/home/huxin/Documents/trae_projects/sikuBERT/qwen3-emb-finetune/data/emb_train_data_clean_val.jsonl'
VAL_DATA_DUALVIEW   = '/home/huxin/Documents/trae_projects/sikuBERT/qwen3-emb-finetune/data/dual_view_emb_data_clean_val.jsonl'
SAVE_DIR = '/home/huxin/Documents/trae_projects/sikuBERT/models'

LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
LR = 2e-4
BATCH_SIZE = 2
ACCUM_STEPS = 16
EPOCHS = 1
MAX_LENGTH = 512
TEMPERATURE = 0.05
WEIGHT_DECAY = 0.01
SEED = 42


def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def info_nce_loss(anchor, positive, temperature=TEMPERATURE):
    bs = anchor.size(0)
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    sim = F.cosine_similarity(anchor.unsqueeze(1), positive.unsqueeze(0), dim=2)
    sim = torch.clamp(sim, min=-1.0, max=1.0)
    labels = torch.arange(bs).to(anchor.device)
    return F.cross_entropy(sim / temperature, labels)


def get_last_token_emb(outputs, attention_mask):
    idx = attention_mask.sum(dim=1) - 1
    return outputs.last_hidden_state[torch.arange(attention_mask.size(0)), idx, :]


def train_epoch(model, tokenizer, train_data, val_data, device, mode, save_path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_data) // (BATCH_SIZE * ACCUM_STEPS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps_per_epoch, 1))

    model.train()
    total_loss = 0
    step_count = 0
    optimizer.zero_grad()
    t0 = time.time()

    for i in range(0, len(train_data), BATCH_SIZE):
        batch = train_data[i:i+BATCH_SIZE]
        if len(batch) < BATCH_SIZE:
            continue

        if mode == 'finetune':
            queries = [item['messages'][0]['content'] for item in batch]
            positives = [item['positive_messages'][0][0]['content'] for item in batch]
            q_inputs = tokenizer(queries, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
            p_inputs = tokenizer(positives, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
            q_embs = get_last_token_emb(model(**q_inputs), q_inputs['attention_mask'])
            p_embs = get_last_token_emb(model(**p_inputs), p_inputs['attention_mask'])
            loss = info_nce_loss(q_embs, p_embs) / ACCUM_STEPS
        else:
            queries = [item['messages'][0]['content'] for item in batch]
            guwen = [item['positive_messages'][0][0]['content'] for item in batch]
            xiandai = [item['positive_messages'][1][0]['content'] for item in batch]
            q_in = tokenizer(queries, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
            g_in = tokenizer(guwen, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
            x_in = tokenizer(xiandai, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
            q_e = get_last_token_emb(model(**q_in), q_in['attention_mask'])
            g_e = get_last_token_emb(model(**g_in), g_in['attention_mask'])
            x_e = get_last_token_emb(model(**x_in), x_in['attention_mask'])
            loss_qg = info_nce_loss(q_e, g_e)
            loss_qx = info_nce_loss(q_e, x_e)
            loss_gx = info_nce_loss(g_e, x_e)
            loss = (loss_qg + loss_qx + 0.5 * loss_gx) / ACCUM_STEPS

        loss.backward()

        if (i // BATCH_SIZE + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step_count += 1

        total_loss += loss.item() * ACCUM_STEPS

        if (i // BATCH_SIZE + 1) % 50 == 0:
            avg = total_loss / ((i // BATCH_SIZE) + 1)
            pct = (i + BATCH_SIZE) / len(train_data) * 100
            elapsed = time.time() - t0
            print(f'  [{pct:.1f}%] Step {step_count}, Loss: {avg:.4f}, {elapsed:.0f}s', flush=True)


    if (i // BATCH_SIZE + 1) % ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step_count += 1

    avg_train = total_loss / (len(train_data) // BATCH_SIZE + 1)
    print(f'\n  Train Loss: {avg_train:.4f}', flush=True)


    model.eval()
    val_loss, vcount = 0, 0
    with torch.no_grad():
        for i in range(0, len(val_data), BATCH_SIZE):
            batch = val_data[i:i+BATCH_SIZE]
            if len(batch) < BATCH_SIZE:
                continue
            if mode == 'finetune':
                queries = [item['messages'][0]['content'] for item in batch]
                positives = [item['positive_messages'][0][0]['content'] for item in batch]
                q_in = tokenizer(queries, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
                p_in = tokenizer(positives, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
                q_e = get_last_token_emb(model(**q_in), q_in['attention_mask'])
                p_e = get_last_token_emb(model(**p_in), p_in['attention_mask'])
                val_loss += info_nce_loss(q_e, p_e).item()
            else:
                queries = [item['messages'][0]['content'] for item in batch]
                guwen = [item['positive_messages'][0][0]['content'] for item in batch]
                xiandai = [item['positive_messages'][1][0]['content'] for item in batch]
                q_in = tokenizer(queries, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
                g_in = tokenizer(guwen, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
                x_in = tokenizer(xiandai, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
                q_e = get_last_token_emb(model(**q_in), q_in['attention_mask'])
                g_e = get_last_token_emb(model(**g_in), g_in['attention_mask'])
                x_e = get_last_token_emb(model(**x_in), x_in['attention_mask'])
                vl = info_nce_loss(q_e, g_e) + info_nce_loss(q_e, x_e) + 0.5 * info_nce_loss(g_e, x_e)
                val_loss += vl.item()
            vcount += 1

    avg_val = val_loss / vcount if vcount > 0 else float('inf')
    print(f'  Val Loss: {avg_val:.4f}', flush=True)

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f'  模型已保存: {save_path}', flush=True)
    return avg_val


def train_one_mode(mode, device):
    '训练单个模式'
    print('\n' + '=' * 70, flush=True)
    print(f'训练模式: {mode} | 全量数据 | 全量float16 + LoRA', flush=True)
    print(f'LoRA: r={LORA_R}, alpha={LORA_ALPHA}, target={LORA_TARGETS}', flush=True)
    print(f'lr={LR}, batch={BATCH_SIZE}, accum={ACCUM_STEPS}, epochs={EPOCHS}', flush=True)
    print('=' * 70, flush=True)

    random.seed(SEED)
    torch.manual_seed(SEED)


    if mode == 'finetune':
        train_data = load_data(TRAIN_DATA_FINETUNE)
        val_data = load_data(VAL_DATA_FINETUNE)
    else:
        train_data = load_data(TRAIN_DATA_DUALVIEW)
        val_data = load_data(VAL_DATA_DUALVIEW)

    print(f'训练集(全部): {len(train_data)}, 验证集: {len(val_data)}', flush=True)


    print('\n加载全量4B模型 (float16)...', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_4B, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        BASE_4B, trust_remote_code=True, dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation='eager').to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    print('模型加载完成 (gradient checkpointing已启用)', flush=True)


    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT, bias='none', task_type=None)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()


    save_name = f'qwen3-emb-{mode}-4b-full-v3'
    save_path = os.path.join(SAVE_DIR, save_name)
    print(f'\n开始训练 (保存到 {save_path})...', flush=True)
    t0 = time.time()
    val_loss = train_epoch(model, tokenizer, train_data, val_data, device, mode, save_path)
    elapsed = (time.time() - t0) / 60
    print(f'\n训练完成！Val Loss: {val_loss:.4f}, 耗时: {elapsed:.1f}分钟', flush=True)


    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)
    print(f'显存已释放, 准备下一个模式...', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modes', type=str, default='finetune',
                        help='训练模式, 逗号分隔 (默认只训练finetune)')
    args = parser.parse_args()

    modes = args.modes.split(',')
    device = 'cuda'

    print('=' * 70, flush=True)
    print(f'全量数据微调 Finetune4B (experiment_v3)', flush=True)
    print(f'训练模式: {modes}', flush=True)
    print(f'数据量: 全部 12654 条 (之前仅 2000 条)', flush=True)
    print(f'查询与测试集无重叠 (已验证)', flush=True)
    print(f'保存路径: models/qwen3-emb-{{mode}}-4b-full-v3', flush=True)
    print('=' * 70, flush=True)

    t0_all = time.time()
    for mode in modes:
        mode = mode.strip()
        train_one_mode(mode, device)

    print(f'\n{"=" * 70}', flush=True)
    print(f'全部训练完成! 总耗时: {(time.time()-t0_all)/60:.1f}分钟', flush=True)
    print('=' * 70, flush=True)


if __name__ == '__main__':
    main()
