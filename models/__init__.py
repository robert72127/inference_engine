from enum import Enum

class MODEL(Enum):
    QWEN_2_5_0_5B = "qwen2_5_0_5b_instruct"

models = {MODEL.QWEN_2_5_0_5B: {"module": "Qwen2_5_0_5B_Instruct", "constructor": "Qwen2_5_0_5B_Instruct"}}

