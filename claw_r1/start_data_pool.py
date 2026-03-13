import ray
from claw_r1.data_pool.data_pool import DataPool, DataPoolConfig
from claw_r1.data_pool.training_backend import VerlBackend
from transformers import AutoTokenizer

NAMESPACE = "claw_r1"  # 与 gateway 保持一致

# 1. 连接到 Ray 集群，并设置命名空间
ray.init(address="auto", namespace=NAMESPACE)

# 2. 配置参数
tokenizer_path = "D:/ai/models/Qwen2.5-0.5B-Instruct"
prompt_length = 4096
response_length = 1024
n_rollouts = 1

# 3. 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4. 创建 backend 和 config
backend = VerlBackend(
    tokenizer=tokenizer,
    prompt_length=prompt_length,
    response_length=response_length
)
config = DataPoolConfig(n_rollouts=n_rollouts)

# 5. 创建 actor，不指定 namespace，让它继承 ray.init 的命名空间
data_pool = DataPool.options(
    name="data_pool",
    lifetime="detached"
).remote(config, backend)

print(f"DataPool actor created with name 'data_pool' in namespace '{NAMESPACE}'")
print("Press Ctrl+C to stop...")
try:
    while True:
        import time
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    ray.kill(data_pool)