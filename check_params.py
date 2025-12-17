from models.model import GPT, GPTConfig

def count_params(config):
    model = GPT(config)
    return model.get_num_params()

configs = {
    "Teacher": GPTConfig(512, 1024, 12, 8, 512),
    "Student 1 (6L, 512E)": GPTConfig(512, 1024, 6, 8, 512),
    "Student 2 (4L, 384E)": GPTConfig(512, 1024, 4, 6, 384),
    "Student 3 (3L, 256E)": GPTConfig(512, 1024, 3, 4, 256),
    "Student 4 (2L, 128E)": GPTConfig(512, 1024, 2, 4, 128),
}

for name, config in configs.items():
    params = count_params(config)
    print(f"{name}: {params/1e6:.2f}M params")
