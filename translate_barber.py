import sys
import os
from datetime import datetime

class BarberPhotoTranslator:
    """Forces the AI to generate a REALISTIC photograph using DSLR keywords."""
    
    def __init__(self, model_name="barber_design_v2"):
        # We use a slightly lower LoRA weight (0.85) to prevent 'deep-frying' the skin texture
        self.lora_trigger = f"<lora:{model_name}:0.85>"
        self.log_file = "/home/kenny/barber_ai/generated_prompts.txt"

    def save_to_file(self, prompt):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"# Generated on {timestamp}\n{prompt}\n\n")
        print(f"[SUCCESS] Photographic Prompt saved.")

    def generate_prompt(self, haircut):
        # Using your exact realism keywords:
        prompt = (
            f"RAW photo, real barbershop photograph, african american male, side profile, "
            f"DSLR 85mm lens, natural lighting, shallow depth of field, "
            f"tight coarse curls, natural hair texture, visible hair strands, "
            f"low taper fade, sharp lineup, {haircut}, "
            f"basketball seams shaved into the fade, curved razor lines forming basketball pattern, "
            f"subtle shaved contrast with visible scalp, "
            f"real skin pores, slight imperfections, natural lighting falloff, "
            f"camera grain, realistic shadows, ultra realistic, high detail, "
            f"{self.lora_trigger}"
        )
        return prompt

if __name__ == "__main__":
    translator = BarberPhotoTranslator()
    
    # Use the first argument as the haircut type, or a default basketball design
    haircut_input = sys.argv[1] if len(sys.argv) > 1 else "curly taper fade with basketball hair design"
    
    final_prompt = translator.generate_prompt(haircut_input)
    print(f"\n--- DSLR PHOTO PROMPT ---\n{final_prompt}\n")
    translator.save_to_file(final_prompt)
