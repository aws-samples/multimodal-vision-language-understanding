import os
import boto3
from langchain import PromptTemplate
from typing import Dict, List
import json
from typing import Optional
import re

# External Dependencies:
from botocore.config import Config

conv_role = {
    'Question': 'human',
    'Answer': 'mixtral'
}

client = boto3.client("sagemaker-runtime", region_name="us-west-2")

def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: Optional[bool] = True,
):
    """Create a boto3 client for Amazon Bedrock, with optional configuration overrides

    Parameters
    ----------
    assumed_role :
        Optional ARN of an AWS IAM role to assume for calling the Bedrock service. If not
        specified, the current active credentials will be used.
    region :
        Optional name of the AWS Region in which the service should be called (e.g. "us-east-1").
        If not specified, AWS_REGION or AWS_DEFAULT_REGION environment variable will be used.
    runtime :
        Optional choice of getting different client to perform operations with the Amazon Bedrock service.
    """
    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]

    if runtime:
        service_name='bedrock-runtime'
    else:
        service_name='bedrock'

    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client

def query_endpoint(payload, llm_endpoint_name):
    try:
        response = client.invoke_endpoint(
            EndpointName=llm_endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
            CustomAttributes="accept_eula=true",
        )
        response = response["Body"].read().decode("utf8")
        return json.loads(response)
    except:
        return None


def format_instructions(instructions: List[Dict[str, str]]) -> List[str]:
    """Format instructions where conversation roles must alternate user/assistant/user/assistant/..."""
    prompt: List[str] = []
    for user, answer in zip(instructions[::2], instructions[1::2]):
        prompt.extend(["<s>", "[INST] ", (user["content"]).strip(), " [/INST] ", (answer["content"]).strip(), "</s>"])
    prompt.extend(["<s>", "[INST] ", (instructions[-1]["content"]).strip(), " [/INST] "])
    return "".join(prompt)


def print_prompt_and_response(prompt: str, response: str) -> None:
    bold, unbold = '\033[1m', '\033[0m'
    print(f"{bold}> Input{unbold}\n{prompt}\n\n{bold}> Output{unbold}\n{response[0]['generated_text']}\n")

def check_qa_pair(qa_pair, image_caption, llm_endpoint_name, attempt_max=5):
    instructions_generator, check_prompt_template = build_conv_instruction_prompt()
    
    instructions_eval=[{"role":"user", "content": check_prompt_template.format(image_caption=image_caption, qa_pairs=qa_pair)}]
    prompt_eval = format_instructions(instructions_eval)
    
    payload = {
        "inputs": prompt_eval,
        "parameters": {"max_new_tokens": 5000, "do_sample": True, "temperature": 0.1}
    }
    
    for i in range(attempt_max):
        new_qa_pair = query_endpoint(payload, llm_endpoint_name)
        if new_qa_pair:
            return new_qa_pair[0]['generated_text']
        print(f'qa check attempt... {i}')
    
    return qa_pair

def parse_qa_response(dataset_text, threshold=4):
    pattern = re.compile(
        r'\d*\.*\s*Question:\s*(.+?)\s*\n*'
        r'\d*\.*\s*Answer:\s*(.+?)\s*\n*'
        r'\d*\.*\s*Rating:\s*(\d+)\n*',
        re.DOTALL
    )
    
    #pattern = re.compile(r'Question: (.*?)\s+Answer: (.*?)\s+Rating: (\d+)\s+Reason: (.*)', re.DOTALL)

    # Find all matches of the pattern
    matches = pattern.findall(dataset_text)

    # Initialize a list to hold the parsed data
    parsed_data = []

    # Iterate over the matches and create a dictionary for each QA block
    for i, match in enumerate(matches):
        question, answer, rating = match
        if int(rating) > threshold:
            # For the first question in each sample, add the <image>\n prefix
            if len(parsed_data) == 0:
                parsed_data.append({"from": "human", "value": "<image>\n" + question.strip()})
            else:
                parsed_data.append({"from": "human", "value": question.strip()})
            # Add the answer to the output_format list
            parsed_data.append({"from": "mixtral", "value": answer})
    return parsed_data

def build_conv_instruction_prompt(prompt_dir="prompts", data_type="conversation"):
    conv_prompt_dir = f"{prompt_dir}/{data_type}"
    
    conv_system_message_path = os.path.join(conv_prompt_dir, "system_message.txt")
    with open(conv_system_message_path, 'r') as f:
        system_message = f.read()
    
    instructions_generator = [{"role":"system", "content": system_message}]
    
    cap_files = sorted([f for f in os.listdir(conv_prompt_dir) if 'caps' in f and f.endswith('_caps.txt')])
    for cap_file in cap_files:
        base_name = cap_file.split('_')[0]  # Assuming the naming convention is 'XXX_caps.txt'
        
        caps_path = os.path.join(conv_prompt_dir, f"{base_name}_caps.txt")
        conv_path = os.path.join(conv_prompt_dir, f"{base_name}_conv.txt")
        
        with open(caps_path, 'r') as f:
            caps = f.read()
        instructions_generator.append({"role":"user", "content": caps})
        
        with open(conv_path, 'r') as f:
            conv = f.read()
        instructions_generator.append({"role":"assistant", "content": conv} )
    
    check_prompt_path = os.path.join(conv_prompt_dir, "check_message.txt")
    with open(check_prompt_path, 'r') as f:
        check_prompt_template = PromptTemplate.from_template(f.read())
    
    return instructions_generator, check_prompt_template

