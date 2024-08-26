#!/usr/bin/env python
# -*- coding: utf-8 -*-

from langchain.schema import (AIMessage, HumanMessage, SystemMessage)
from langchain_openai import ChatOpenAI

# export OPENAI_API_KEY="sk-xxxx"
# export OPENAI_API_BASE="https://api.openai.com/v1"

class LLMs:
    def __init__(self, model="gpt-4-1106-preview", request_type="openai", parameters={"top_p": 0.7, "temperature": 0.9}, is_greedy=False):
        self.model = model
        self.request_type = request_type

        assert request_type == "openai"
        
        if is_greedy:
            print("Using greedy decoding")
            parameters["top_p"] = 1.0
            parameters["temperature"] = 0.0
        
        self.client = ChatOpenAI(model_name=model)
        self.client.model_kwargs = parameters

    def request(self, prompt, system_message=None):
        try:
            batch_messages = [[
                HumanMessage(content=prompt),
            ] if system_message is None else [
                SystemMessage(content=system_message),
                HumanMessage(content=prompt),
            ]]

            results = self.client.generate(batch_messages)
            model_output = results.generations[0][0].text
            return model_output
        except Exception as e:
            print(e)
            return None

if __name__ == "__main__":
    pass