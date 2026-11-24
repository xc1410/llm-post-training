import json
import re
import time
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

with open('data/gsm8k_test.jsonl',encoding='utf-8') as f:
    test_rows=[json.loads(line)for line in f]
print(len(test_rows))
print(test_rows[0]['question'])
print(test_rows[0]['answer'])
assert len(test_rows)==1319,len(test_rows)

parts=test_rows[0]['answer'].split('####')
print(len(parts))
print(parts)
print(parts[-1])

gold=parts[-1].strip()
print(gold)
print(gold=='18',gold==18)
assert gold=='18',gold

def gold_answer(answer_text):
    return answer_text.split('####')[-1].strip().replace(',','')

golds=[gold_answer(row['answer'])for row in test_rows]
print(golds[:10],golds[146])
assert golds[0]=='18' and golds[146]=='2125',(golds[0],golds[146])

def extract_answer(text):
    boxed=re.findall(r'\\boxed\{([^}]*)\}',text)
    if boxed:
        text=boxed[-1]
    numbers=re.findall(r'-?\d+(?:\.\d+)?',text.replace(',',''))
    if not numbers:
        return None
    return numbers[-1]

def is_correct(pred,gold):
    return pred!=None and abs(float(pred)-float(gold))<1e-6

print(extract_answer('He paid $5 for 3 apples.'))
assert extract_answer('So she makes 18 dollars.')=='18' and extract_answer('Total: 2,125.')=='2125'
assert extract_answer('It drops to -3 degrees.')=='-3' and extract_answer('I do not know.')==None
print(extract_answer('The answer is \\boxed{18}. This took 2 steps.'))
assert extract_answer('The answer is \\boxed{18}. This took 2 steps.')=='18'
assert is_correct('3.50','3.5') and not is_correct(None,'18') and not is_correct('17','18')

model_dir='/data/xc26/models/Qwen2.5-0.5B-Instruct'
tokenizer=AutoTokenizer.from_pretrained(model_dir)
SYSTEM='You are a helpful assistant.'
SUFFIX='\nPlease reason step by step, and put your final answer within \\boxed{}.'

def build_prompt(question):
    messages=[{'role':'system','content':SYSTEM},{'role':'user','content':question+SUFFIX}]
    return tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

print(build_prompt(test_rows[0]['question']))
print(len(tokenizer(build_prompt(test_rows[0]['question']))['input_ids']))

device='cuda' if torch.cuda.is_available() else 'cpu'
model=AutoModelForCausalLM.from_pretrained(model_dir,dtype=torch.float32).to(device)

def generate_answer(question):
    inputs=tokenizer(build_prompt(question),return_tensors='pt').to(device)
    out_ids=model.generate(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'],max_new_tokens=512,do_sample=False,repetition_penalty=1.0,eos_token_id=[151645,151643])
    return tokenizer.decode(out_ids[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)

response=generate_answer(test_rows[0]['question'])
print(response)
print(extract_answer(response),golds[0],is_correct(extract_answer(response),golds[0]))

results=[]
start=time.perf_counter()
for row in test_rows[:20]:
    response=generate_answer(row['question'])
    pred=extract_answer(response)
    gold=gold_answer(row['answer'])
    results.append({'question':row['question'],'response':response,'pred':pred,'gold':gold,'correct':is_correct(pred,gold)})
print('用时',time.perf_counter()-start)
print('准确率',sum(r['correct'] for r in results)/len(results))

with open('outputs/instruct_test20.jsonl','w',encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r,ensure_ascii=False)+'\n')