import json, random, os, time, gc, argparse, sys
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

os.environ['TRITON_DISABLE_CUDA_KERNEL'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


ROOT_DIR = os.environ.get('CCRE_LVF_ROOT', 'path/to/project')
BASE_MODEL = os.path.join(ROOT_DIR, 'models', 'base-model')
TRAIN_DATA = os.path.join(ROOT_DIR, 'data', 'train.jsonl')
VAL_DATA = os.path.join(ROOT_DIR, 'data', 'validation.jsonl')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'models', 'adapted-model')

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


def train_epoch(model, tokenizer, train_data, val_data, device, save_path):
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

        queries = [item['messages'][0]['content'] for item in batch]
        positives = [item['positive_messages'][0][0]['content'] for item in batch]
        q_inputs = tokenizer(queries, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
        p_inputs = tokenizer(positives, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
        q_embs = get_last_token_emb(model(**q_inputs), q_inputs['attention_mask'])
        p_embs = get_last_token_emb(model(**p_inputs), p_inputs['attention_mask'])
        loss = info_nce_loss(q_embs, p_embs) / ACCUM_STEPS

        loss.backward()

        if (i // BATCH_SIZE + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step_count += 1

        total_loss += loss.item() * ACCUM_STEPS

    
    if (i // BATCH_SIZE + 1) % ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step_count += 1

    avg_train = total_loss / (len(train_data) // BATCH_SIZE + 1)


    
    model.eval()
    val_loss, vcount = 0, 0
    with torch.no_grad():
        for i in range(0, len(val_data), BATCH_SIZE):
            batch = val_data[i:i+BATCH_SIZE]
            if len(batch) < BATCH_SIZE:
                continue
            queries = [item['messages'][0]['content'] for item in batch]
            positives = [item['positive_messages'][0][0]['content'] for item in batch]
            q_in = tokenizer(queries, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
            p_in = tokenizer(positives, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').to(device)
            q_e = get_last_token_emb(model(**q_in), q_in['attention_mask'])
            p_e = get_last_token_emb(model(**p_in), p_in['attention_mask'])
            val_loss += info_nce_loss(q_e, p_e).item()
            vcount += 1

    avg_val = val_loss / vcount if vcount > 0 else float('inf')


    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    return avg_val


def train_encoder(base_model, train_path, val_path, output_path, device):
    random.seed(SEED)
    torch.manual_seed(SEED)
    train_data = load_data(train_path)
    val_data = load_data(val_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        base_model, trust_remote_code=True, dtype=torch.float16,
        low_cpu_mem_usage=True, attn_implementation='eager').to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=LORA_DROPOUT, bias='none', task_type=None)
    model = get_peft_model(model, lora_config)
    val_loss = train_epoch(model, tokenizer, train_data, val_data, device, output_path)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', default=BASE_MODEL)
    parser.add_argument('--train-data', default=TRAIN_DATA)
    parser.add_argument('--validation-data', default=VAL_DATA)
    parser.add_argument('--output-dir', default=OUTPUT_DIR)
    args = parser.parse_args()
    val_loss = train_encoder(args.base_model, args.train_data, args.validation_data, args.output_dir, 'cuda')
    print(json.dumps({'validation_loss': val_loss}, ensure_ascii=False))


if __name__ == '__main__':
    main()
