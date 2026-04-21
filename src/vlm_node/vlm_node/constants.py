import os

# ---- VLM API keys ----
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")

# ---- Provider selection ----
# Explicit via VLM_PROVIDER=gemini|qwen, or auto-detected from which key is set.
# If both keys are set and VLM_PROVIDER is unset, Gemini wins (preserves prior behavior).
VLM_PROVIDER = os.environ.get("VLM_PROVIDER", "").lower()
if VLM_PROVIDER not in ("gemini", "qwen"):
    if GEMINI_API_KEY:
        VLM_PROVIDER = "gemini"
    elif DASHSCOPE_API_KEY:
        VLM_PROVIDER = "qwen"
    else:
        VLM_PROVIDER = "gemini"  # will fail at connect time; surfaces missing-key error clearly

if VLM_PROVIDER == "qwen":
    VLM_API_KEY = DASHSCOPE_API_KEY
    VLM_BASE_URL = "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
    MODEL_NAME = os.environ.get("QWEN_MODEL", "qwen3.6-plus")
    MODEL_NAME_LITE = os.environ.get("QWEN_MODEL_LITE", "qwen3.6-flash")
else:
    VLM_API_KEY = GEMINI_API_KEY
    VLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    MODEL_NAME = "gemini-2.5-flash"
    MODEL_NAME_LITE = "gemini-2.5-flash-lite"

# Target object to search for in the scene
target_object = ""
room_condition     = ""
spatial_condition  = ""
anchor_object      = ""
attribute_condition = ""
