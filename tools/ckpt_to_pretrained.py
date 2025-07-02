import torch
from models import TimesformerForMultiTaskingSigLIP, StreamformerConfig

def ckpt_to_pretrained(ckpt_path, initial_model_name, pretrained_model_name):
    ckpt = torch.load(ckpt_path)
    multi_task_config = {
        "THUMOS14": {
            "label2id": "RESERVED"
        }
    }
    model = TimesformerForMultiTaskingSigLIP.from_pretrained(initial_model_name, multi_task_config, ignore_mismatched_sizes=True)
    model.load_state_dict(ckpt['model'])
    model.save_pretrained(pretrained_model_name)
    
def main():
    ckpt_to_pretrained("checkpoints/siglip_multi_task_thumos14/pretrained")

if __name__ == "__main__":
    main()