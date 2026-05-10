from enum import Enum

class MODEL(Enum):
    QWEN_2_5_0_5B_INSTRUCT = "qwen2_5_0_5b_instruct"

models = {MODEL.QWEN_2_5_0_5B_INSTRUCT: {"module": "Qwen2_5_0_5B_instruct", "constructor": "Qwen2_5_0_5B_Instruct"}}

