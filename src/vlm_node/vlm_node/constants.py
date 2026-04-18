import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

MODEL_NAME = 'gemini-2.5-flash'

# Target object to search for in the scene
target_object = ""
room_condition     = ""
spatial_condition  = ""
anchor_object      = ""
attribute_condition = ""