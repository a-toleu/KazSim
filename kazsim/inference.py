from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch
from typing import Optional, List

class SimplificationInference:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    
    def prepare_for_inference(self):
        """Enable faster inference mode."""
        FastLanguageModel.for_inference(self.model)
    
    def simplify_text(
        self,
        text: str,
        instruction: str = "Simplify the following text",
        max_new_tokens: int = 256,
        stream: bool = False
    ) -> str:
        """Simplify a given text."""
        self.prepare_for_inference()
        
        prompt = self.prompt_template.format(
            instruction,
            text,
            ""  # Leave output blank for generation
        )
        
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        
        if stream:
            text_streamer = TextStreamer(self.tokenizer)
            outputs = self.model.generate(
                **inputs, 
                streamer=text_streamer, 
                max_new_tokens=max_new_tokens
            )
        else:
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded.split("### Response:")[-1].strip()
        
        return response
    
    def batch_simplify(
        self,
        texts: List[str],
        instruction: str = "Simplify the following text",
        max_new_tokens: int = 256
    ) -> List[str]:
        """Simplify multiple texts."""
        results = []
        for text in texts:
            simplified = self.simplify_text(text, instruction, max_new_tokens, stream=False)
            results.append(simplified)
        return results