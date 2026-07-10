import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "microsoft/Phi-3-mini-4k-instruct"

# VRAM 절약을 위한 4비트 양자화 설정
# 하드웨어에 따라 적절한 compute dtype 및 4-bit 사용 여부 결정
has_cuda = torch.cuda.is_available()
try:
    bf16_supported = torch.cuda.is_bf16_supported()
except Exception:
    bf16_supported = False

use_4bit = has_cuda  # 4-bit 양자화는 보통 GPU에서 사용
compute_dtype = torch.bfloat16 if (has_cuda and bf16_supported) else torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype
)

# 토크나이저 및 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_id)
if has_cuda:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
else:
    # GPU가 없으면 4-bit 양자화를 사용하지 않고 CPU로 로드
    if use_4bit:
        bnb_config.load_in_4bit = False
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",
    )

from peft import LoraConfig, TaskType

peft_config = LoraConfig(
    r=8,                  # 가중치 행렬의 차원 (작을수록 메모리 절약)
    lora_alpha=16,        # 스케일링 계수 (보통 r의 2배)
    lora_dropout=0.05,
    bias="none",
    # target_modules는 모델 아키텍처에 따라 달라집니다. 일반적인 causal LM용 예시:
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type=TaskType.CAUSAL_LM # 인과적 언어 모델링 지정
)

from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# 샘플 데이터셋 로드
dataset = load_dataset("HuggingFaceTB/smoltalk","all", split="train")
# 간단한 토크나이징: SFTConfig는 `max_seq_length`를 받지 않을 수 있으므로
# 토큰화 단계에서 `max_length`로 시퀀스 길이를 제한합니다.
def _extract_text_from_value(v):
    # v can be str, list[str], list[dict], dict, etc.
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        # try common text keys
        for k in ("text", "content", "utterance", "message"):
            if k in v and isinstance(v[k], str):
                return v[k]
        # otherwise join string values
        parts = [str(x) for x in v.values() if isinstance(x, str)]
        return " ".join(parts) if parts else ""
    if isinstance(v, list):
        # list of strings
        if all(isinstance(x, str) for x in v):
            return " ".join(v)
        # list of dicts
        if all(isinstance(x, dict) for x in v):
            parts = []
            for item in v:
                for k in ("text", "content", "utterance", "message"):
                    if k in item and isinstance(item[k], str):
                        parts.append(item[k])
                        break
            return " ".join(parts)
    return ""

def tokenize_example(examples):
    # examples is a dict of lists when batched=True
    # find a candidate text field (list of strings or list of dicts)
    text_key = None
    for k, v in examples.items():
        if isinstance(v, list) and len(v) > 0:
            # check first element to decide
            first = v[0]
            if isinstance(first, str) or isinstance(first, dict):
                text_key = k
                break
    if text_key is None:
        # fallback: try to build text from all fields
        texts = []
        for i in range(len(next(iter(examples.values())))):
            parts = []
            for v in examples.values():
                parts.append(_extract_text_from_value(v[i]))
            texts.append(" ".join([p for p in parts if p]))
    else:
        vals = examples[text_key]
        texts = []
        for v in vals:
            text = _extract_text_from_value(v)
            texts.append(text)

    tokenized = tokenizer(texts, truncation=True, max_length=512)
    return tokenized

# Use batched tokenization and remove original columns
dataset = dataset.map(tokenize_example, batched=True, remove_columns=dataset.column_names)
dataset.set_format(type="torch")

# 학습 세부 설정 (지원되는 인자들만 전달)
training_args = SFTConfig(
    output_dir="./my_finetuned_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=1,
    use_cpu=not has_cuda,
)

# 트레이너 선언 및 학습 시작
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
)

# Some SFTTrainer versions don't accept `tokenizer` in the constructor;
# attach it if the trainer accepts the attribute after construction.
if not hasattr(trainer, "tokenizer"):
    try:
        trainer.tokenizer = tokenizer
    except Exception:
        pass

trainer.train()