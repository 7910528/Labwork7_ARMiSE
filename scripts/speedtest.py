import sys, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
name, rev, dtype = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(2)
t0=time.time()
tok=AutoTokenizer.from_pretrained(name, revision=rev)
model=AutoModelForCausalLM.from_pretrained(name, revision=rev, dtype=getattr(torch,dtype))
model.eval()
print("load", round(time.time()-t0,1),"s")
msgs=[{"role":"user","content":'Implement the following task in Python: "Write a python function to find the first repeated character in a given string.". The function must be named first_repeated_char. Return only the function code.'}]
ids=tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
torch.manual_seed(0)
t0=time.time()
out=model.generate(**ids, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.95)
n=out.shape[1]-ids["input_ids"].shape[1]
dt=time.time()-t0
print("gen", n, "tokens", round(dt,1), "s", round(n/dt,1), "tok/s")
print(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True))
