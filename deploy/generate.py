from download_manager import download_bucket_content
import transformers
import torch

download_bucket_content(bucket_name="finetune_v1",  local_folder="model")

class GemmaGenerate:
    def __init__(self) -> None:
        self.__model = transformers.GemmaForCausalLM.from_pretrained(
            "model",
            local_files_only=True,
            device_map = "auto"
        )

        self.__tokenizer = transformers.GemmaTokenizer.from_pretrained(
            "model",
            local_files_only=True,
        )

    def generate(self, input:str) -> str:
        inputs = self.__tokenizer([input], return_tensors="pt").to(self.__model.device)
        with torch.no_grad():
            if not torch.backends.mps.is_available():
                with torch.amp.autocast(device_type="cuda"):
                    outputs = self.__model.generate(**inputs, max_length=150, 
                                                    do_sample=False, 
                                                    return_dict_in_generate=True)
            else:
                with torch.amp.autocast(device_type="mps"):
                    #torch.mps.profiler.start(mode="interval,event")
                    print("start generating tokens")
                    outputs = self.__model.generate(**inputs, max_length=150, 
                                                    do_sample=False, 
                                                    return_dict_in_generate=True)
                    print("done generating tokens")
                    #torch.mps.profiler.stop()
        generated_text = ""
        for token in outputs.sequences[0]:

            decoded_token = self.__tokenizer.decode([token], skip_special_tokens=True)
            generated_text += decoded_token
            print(decoded_token, end="", flush=True)
        return generated_text
    


gg = GemmaGenerate()
rst = gg.generate("As a healthcare fellow learning diagnosis, What to do for Henoch-Schnlein Purpura")
#print(rst)