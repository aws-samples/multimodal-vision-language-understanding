import os
import argparse
import time
from utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed

coco_instruct_map = {
    'conversation': 'conversation_58k.json',
    'detail': 'detail_23k.json',
    'complex_reasoning': 'complex_reasoning_77k.json'
}

img_digit_num = {
    'coco': 12
}

def load_coco(coco_caption_path):
    with open(coco_caption_path, 'r') as f:
        coco_caption = json.load(f)

    coco_cap_new = {}
    for cap in coco_caption['annotations']:
        if cap['image_id'] in coco_cap_new:
            coco_cap_new[cap['image_id']].append(cap['caption'])
        else:
            coco_cap_new[cap['image_id']] = [cap['caption']]
            
    return coco_cap_new

def filter_images(llava_instruct_data_path, coco_caps, dataset_name='coco'):
    with open(llava_instruct_data_path, 'r') as f:
        llava_instruct_list = json.load(f)
    
    llava_instruct_dict = {}
    #turn_num = []
    for ins_data in llava_instruct_list:
        if ins_data['id'] in llava_instruct_dict:
            llava_instruct_dict[ins_data['id']].append(ins_data)
        else:
            llava_instruct_dict[ins_data['id']] = [ins_data]
            #turn_num.append(len(ins_data['conversations'])/2)
    
    filtered_data = {}
    for img_id, ann in llava_instruct_dict.items():
        if 'image' in ann[0] and dataset_name in ann[0]['image']:
            filtered_data[str(int(img_id))] = ann
    coco_ids = [img_id for img_id in filtered_data.keys()]
    
    return {coco_id: coco_caps[int(coco_id)] for coco_id in coco_ids if int(coco_id) in coco_caps}

def generate_answers_for_image(data, digit_num, attempt_max=5):
    img_id, cap, llm_endpoint_name, instructions_generator = data

    print(f'processing image-{img_id} ...')
    # Generate answers
    start_time = time.time()
    image_caption = '\n'.join(cap)
    instruct_data = instructions_generator.copy()
    instruct_data.append({"role": "user", "content": image_caption})

    payload = {
        "inputs": format_instructions(instruct_data),
        "parameters": {"max_new_tokens": 5000, "do_sample": True}
    }
    
    for i in range(attempt_max):
        response = query_endpoint(payload, llm_endpoint_name)
        if response:
            break
        print(f'qa generation attempt... {i}')

    if response is None:
        print('query failed!!')
        return None
    
    qa_pair = response[0]['generated_text']
    
    filtered_qa = check_qa_pair(qa_pair, image_caption, llm_endpoint_name, attempt_max=attempt_max) 
    
    if filtered_qa is None:
        print('check failed!!')
        return None
    img_id = img_id.zfill(digit_num)
    
    conv_data = {
        "id": img_id,
        "image": f"coco/train2017/{img_id}.jpg",
        "conversations": parse_qa_response(filtered_qa)
    }
    
    print(f'{img_id} inference duration is {time.time() - start_time}')
    return img_id, cap, conv_data

def data_gen(coco_caps, llm_endpoint_name, dataset_path, digit_num=12, caps_coco_path=None, max_workers=5):
    cap_list = {}
    conv_list = []

    instructions_generator, _ = build_conv_instruction_prompt()
    
    sample_num = 1000
    # Prepare data for multithreading
    data_for_multithreading = [(img_id, cap, llm_endpoint_name, instructions_generator) for img_id, cap in coco_caps.items()][:sample_num]

    # Use ThreadPoolExecutor to generate answers in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_img_id = {executor.submit(generate_answers_for_image, data, digit_num): data[0] for data in data_for_multithreading}
        
        for idx, future in enumerate(as_completed(future_to_img_id)):
            print(f'idx: {idx}')
            result = future.result()
            print(f'Processing... {idx*100/sample_num}% {idx}/{sample_num}')
            if result is not None:
                img_id, cap, conv_data = result
                conv_list.append(conv_data)
                cap_list[img_id] = cap
    
    # Save the conversation data
    with open(dataset_path, 'w') as f:
        json.dump(conv_list, f, indent=4)
    
    # Optionally save the captions data
    if caps_coco_path:
        with open(caps_coco_path, 'w') as f:
            json.dump(cap_list, f, indent=4)

def main(args):
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    if args.dataset == 'coco':
        coco_caption = load_coco(args.coco_caption_path)
        filtered_coco_caps = filter_images(os.path.join(args.llava_instruct_dir, 'llava_v1_5_mix665k.json'), coco_caption, dataset_name=args.dataset)
        dataset_path = os.path.join(args.save_dir, 'conversation_58k.json')
        caps_coco_path = os.path.join(args.save_dir, 'coco_caps.json')
        data_gen(filtered_coco_caps, args.llm_endpoint, dataset_path, digit_num=img_digit_num[args.dataset], caps_coco_path=caps_coco_path, max_workers=args.max_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument('--llm-endpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="coco")
    parser.add_argument('--coco-caption-path', type=str, default="dataset/annotations/captions_train2017.json")
    parser.add_argument('--llava-instruct-dir', type=str, default="./dataset/LLaVA-Instruct-150K")
    parser.add_argument('--save-dir', type=str, default="./dataset/mixtral_coco")
    parser.add_argument('--max-workers', type=int, default=5) # set the number of endpoints to max_workers so that each endpoint can handle query from one worker.
    
    args = parser.parse_args()
    main(args)
