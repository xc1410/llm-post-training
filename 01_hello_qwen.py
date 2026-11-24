import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

model_dir='/data/xc26/models/Qwen2.5-0.5B-Instruct'
tokenizer=AutoTokenizer.from_pretrained(model_dir)
ids=tokenizer('hello world')['input_ids']
print(ids)
print(tokenizer.decode(ids))

sentence='Natalia sold clips to 48 of her friends in April.'
ids=tokenizer(sentence)['input_ids']
print(len(ids),ids)
print([tokenizer.decode([i])for i in ids])

messages=[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':'1+1等于几？'}]
text=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
print(text)

chat_ids=tokenizer(text)['input_ids']
print(len(chat_ids),chat_ids)
print(tokenizer.eos_token,tokenizer.eos_token_id)

device='cuda' if torch.cuda.is_available() else 'cpu'
model=AutoModelForCausalLM.from_pretrained(model_dir,dtype=torch.float32).to(device)
n_params=sum(p.numel() for p in model.parameters())
assert n_params==494032768,n_params
print(model)

inputs=tokenizer(text,return_tensors='pt').to(device)
with torch.no_grad():
    out=model(inputs['input_ids'])
print(inputs['input_ids'].shape,out.logits.shape)
print(tokenizer.decode(out.logits[0,-1].argmax().item()))

print(model.generation_config)
out_ids=model.generate(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],max_new_tokens=64,do_sample=False,repetition_penalty=1.0)
new_ids=out_ids[0,inputs['input_ids'].shape[1]:]
print(out_ids.shape,tokenizer.decode(new_ids,skip_special_tokens=True))

ids_t=inputs['input_ids']
with torch.no_grad():
    for _ in range(64):
        next_t=model(ids_t).logits[:,-1,:].argmax(dim=1,keepdim=True)
        ids_t=torch.cat([ids_t,next_t],dim=1)
        if next_t.item()==tokenizer.eos_token_id:
            break
assert ids_t.shape==out_ids.shape and (ids_t==out_ids).all(),(ids_t.shape,out_ids.shape)