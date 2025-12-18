import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Llama 3.2 3B Inference")
    parser.add_argument("--quantize", action="store_true", help="Enable 4-bit quantization (Phase 2)")
    args = parser.parse_args()

    print("=== Phase 1: System Verification ===")
    
    # 1. Strict CUDA Check
    if not torch.cuda.is_available():
        print("❌ CRITICAL ERROR: CUDA is NOT available.")
        print("   This script requires a GPU. Please reinstall PyTorch with CUDA support.")
        print("   Run: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)
    
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    
    print(f"\n=== Phase 2: Model Loading (Mode: {'4-bit Quantization' if args.quantize else 'FP16'}) ===")
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    
    try:
        print(f"Loading tokenizer for {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if args.quantize:
            print("Configuring 4-bit quantization (bitsandbytes)...")
            # Phase 2: 4-bit Quantization Config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print(f"Loading model {model_id} in 4-bit...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="cuda", # Explicitly set to cuda for bitsandbytes
            )
        else:
            print(f"Loading model {model_id} in float16...")
            # Phase 1: FP16 (half precision)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16,
                device_map="auto"
            )
        
        model.eval() # Ensure evaluation mode
        
    except Exception as e:
        print(f"\n❌ ERROR Loading Model: {e}")
        print("   Common causes:")
        print("   1. 401 Unauthorized -> You didn't accept the license or login.")
        print("   2. OOM -> Your GPU ran out of memory.")
        if args.quantize:
            print("   3. ImportError -> 'bitsandbytes' not installed. Run 'pip install bitsandbytes'.")
        sys.exit(1)

    print("\n=== Phase 3: Device Verification ===")
    # Strict verification that model is on GPU
    model_device = model.device
    param_device = next(model.parameters()).device
    
    print(f"✅ Model Device: {model_device}")
    print(f"✅ Parameter Device: {param_device}")
    
    if "cuda" not in str(model_device) and "cuda" not in str(param_device):
        print("❌ ERROR: Model is NOT on CUDA. It is running on CPU!")
        sys.exit(1)

    print("\n=== Phase 4: Inference Test ===")
    
    # FIX: Ensure tokenizer has a pad_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    prompt = "Explain the concept of 'entropy' in one sentence."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    
    print(f"Input Prompt: {prompt}")
    print("Generating response...")
    
    # FIX: Use tokenize=False to get the formatted string first
    prompt_str = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # FIX: Tokenize the string to get both input_ids and attention_mask automatically
    inputs = tokenizer(
        prompt_str,
        return_tensors="pt",
        truncation=True
    ).to(model.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    with torch.no_grad():
        # FIX: Explicitly unset generation flags that conflict with do_sample=False
        # This prevents "generation flags are not valid" warnings
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask, # FIX: Pass tokenizer-generated mask
            pad_token_id=tokenizer.pad_token_id,  # FIX: Pass pad token ID
            max_new_tokens=100,
            eos_token_id=terminators,
            do_sample=False, # Deterministic generation
        )
        
    response = outputs[0][inputs.input_ids.shape[-1]:]
    decoded_response = tokenizer.decode(response, skip_special_tokens=True)
    
    print("\n=== ✅ SUCCESS: Generated Response ===")
    print(decoded_response)
    print("======================================")

if __name__ == "__main__":
    main()
