prompt_wo_context = """Evaluate the responses of AI models to the following multiple-choice question in the field of bio-medical.

## Question and Reference Answer
Question: {question}

Reference Answer: {answer}

## Model Responses
{candidates}

## Evaluation Criteria
Using the criteria of Helpfulness, Faithfulness, and Verifiability, provide detailed feedback for each model's response. Consider the following in your evaluation:
- Helpfulness: How effectively does the response address the core question?
- Faithfulness: How accurately does the response reflect the correct answer and factual context?
- Verifiability: Can the response's claims be easily supported by evidence?

## Feedback and Rankings
For each response, identify strengths, areas for improvement, and provide an overall score between 0 to 10 (where 10 is the highest). Conclude with a ranking of the model responses based on their adherence to these criteria.

Format your feedback and rankings as follows:

```
{{
  "feedback": {{
    "Model 1": {{
      "Helpfulness": "",
      "Faithfulness": "",
      "Verifiability": "",
      "Overview Score": ""
    }},
    // Similar entries for other models
  }},
  "ranking": [
    {{"rank": 1, "model": "Model X"}},
    // Subsequent rankings
  ]
}}
```
"""

prompt_sketch_start = """Please evaluate the following user instruction and the proposed response within the context of biomedicine.

## Evaluation Criteria
Use the following 5-point scale to assess how well the AI Assistant's response addresses the biomedical inquiry:

1: Inadequate - The response is incomplete, vague, off-topic, or controversial. It may lack necessary biomedical data, use incorrect terminology, or include irrelevant clinical examples. The perspective may be inappropriate, such as personal experiences from non-scientific blogs or resembling a forum answer, which is unsuitable given the precision required in biomedicine.

2: Partially Adequate - The response addresses most biomedical aspects requested but lacks direct engagement with the core scientific question. It might provide a general overview instead of detailed biomedical mechanisms or specific clinical applications.

3: Acceptable - The response is helpful, covering all basic biomedical queries. However, it may not adopt an AI Assistant’s typical scientific voice, resembling content from general health blogs or web pages and could include personal opinions or generic information.

4: Good - The response is clearly from an AI Assistant, accurately focusing on the biomedical instruction. It is complete, clear, and comprehensive, presented in a clinically appropriate tone. Minor improvements could include adding more precise scientific details or a more formal presentation.

5: Excellent - The response perfectly represents an AI Assistant in biomedicine, addressing the user's scientific inquiry without any irrelevant content. It demonstrates in-depth knowledge, is scientifically accurate, logically structured, engaging, insightful, and impeccably written.

"""

prompt_sketch_end = """
## Feedback and Rankings
Provide feedback and an overall score between 1 to 5 for each response based on the **Evaluation Criteria**. Then rank the model responses, even if they share the same score, based on criteria such as clarity of response logic, richness of information, and naturalness of language.

Format your feedback and rankings as follows:

```
{{
  "feedback": {{
    "Model 1": {{
      "Evaluation": "",
      "Score": ""
    }},
    // Similar entries for other models
  }},
  "ranking": [
    {{"rank": 1, "model": "Model X"}},
    // Subsequent rankings
  ]
}}
```
"""

prompt_wo_context_mid = """## Question and Reference Answer
Question: {question}

Reference Answer: {answer}

## Model Responses
{candidates}

"""

prompt_wo_context_open_mid = """## Question
{question}

## Model Responses
{candidates}
"""

prompt_wo_context_new = prompt_sketch_start + prompt_wo_context_mid + prompt_sketch_end
prompt_wo_context_open_new = prompt_sketch_start + prompt_wo_context_open_mid + prompt_sketch_end

prompt_wo_context_open = """Evaluate the responses of AI models to the following bio-medical questions.

## Question
{question}

## Model Responses
{candidates}

## Evaluation Criteria
Using the criteria of Helpfulness, Faithfulness, and Verifiability, provide detailed feedback for each model's response. Consider the following in your evaluation:
- Helpfulness: How effectively does the response address the core question?
- Faithfulness: How accurately does the response reflect the correct answer and factual context?
- Verifiability: Can the response's claims be easily supported by evidence?

## Feedback and Rankings
For each response, identify strengths, areas for improvement, and provide an overall score between 0 to 10 (where 10 is the highest). Conclude with a ranking of the model responses based on their adherence to these criteria.

Format your feedback and rankings as follows:

```
{{
  "feedback": {{
    "Model 1": {{
      "Helpfulness": "",
      "Faithfulness": "",
      "Verifiability": "",
      "Overview Score": ""
    }},
    // Similar entries for other models
  }},
  "ranking": [
    {{"rank": 1, "model": "Model X"}},
    // Subsequent rankings
  ]
}}
```
"""


prompt_wo_context_optima = """Evaluate the responses of AI models to the following multiple-choice question in the field of bio-medical.

## Question and Reference Answer
Question: {question}

Reference Answer: {answer}

## Model Responses
{candidates}

## Evaluation Criteria
Using the criteria of Helpfulness, Faithfulness, and Verifiability, provide detailed feedback for each model's response. Consider the following in your evaluation:
- Helpfulness: How effectively does the response address the core question?
- Faithfulness: How accurately does the response reflect the correct answer and factual context?
- Verifiability: Can the response's claims be easily supported by evidence?

## Feedback and Rankings
For each response, identify strengths, areas for improvement, and provide an overall score between 0 to 10 (where 10 is the highest). Conclude with a ranking of the model responses based on their adherence to these criteria.

## Optimized Response
Based on the evaluations, craft a detailed and comprehensive final answer that fully addresses the question, integrating insights from the evaluation process. Select the top response as a starting point for optimization. If the top-ranked response has any shortcomings in accuracy or completeness, revise it to enhance clarity, depth, and completeness. If necessary, supplement it with additional information to ensure it meets the highest standards of Helpfulness, Faithfulness, and Verifiability. The final answer should not only select the correct option but also explain the reasoning behind the choice in detail, as seen in the best model responses.

## Output Format
Format your feedback, rankings, and the modified answer as follows:

```
{{
  "feedback": {{
    "Model 1": {{
      "Helpfulness": "",
      "Faithfulness": "",
      "Verifiability": "",
      "Overview Score": ""
    }},
    // Similar entries for other models
  }},
  "ranking": [
    {{"rank": 1, "model": "Model X"}},
    // Subsequent rankings
  ],
  "Final Answer": "<Provide a detailed and comprehensive explanation that fully addresses the question, with step-by-step analysis and justification. Incorporate any necessary corrections or additional insights from the best model response to achieve an overall score of 10. Ensure the answer reflects the depth and detail observed in the best model responses. Conclude with 'Thus, the correct option is X,' where X is the correct answer (A, B, C, or D).>"
}}
```
"""

prompt_w_context = """Assess AI model responses to a specific question, using provided context as a grounding text for a more informed evaluation. This context, not seen by the models, will help you discern the accuracy and relevance of their responses. Your evaluation will critically determine how well each response aligns with knowledgeable insights and established facts related to the topic.

## Context for Evaluation
{context}

## Question and Reference Answer
Question: {question}

Correct Answer: {answer}

## Model Responses
{candidates}

## Evaluation Criteria
Assessment Criteria in Light of Context:
- Helpfulness: Does the response effectively inform on the core question, considering the grounding context?
- Faithfulness: Is the response accurate and faithful to the correct answer, as supported by the context?
- Verifiability: Within the context's framework, can the response's assertions be easily corroborated?

## Feedback and Rankings
Evaluate each model's response against these criteria, highlighting strengths and suggesting areas for improvement. Provide an overall score from 0 to 10 for each model, where 10 signifies the highest achievement in the evaluation criteria. Conclude with a ranking of the responses based on their overall adherence to these criteria.

Feedback and Rankings Format:

```
{
  'feedback': {
    'Model 1': {
      'Helpfulness': '',
      'Faithfulness': '',
      'Verifiability': '',
      'Overview Score': ''
    },
    // Entries for other models follow
  },
  'ranking': [
    {'rank': 1, 'model': 'Model X'},
    // Followed by subsequent rankings
  ]
}
```

Note: Use the context as a reference point to ground your evaluation, ensuring a nuanced and informed analysis. This approach emphasizes the importance of aligning the model responses with established knowledge and insights on the topic.
"""

# judge_pair_json = {"system_prompt": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.", "prompt_template": "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"}
judge_pair_json = {
  "system_prompt": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the biomedical question displayed below. You should choose the assistant that best follows the user's instructions and answers the question accurately and thoroughly. Your evaluation should consider factors such as helpfulness, truthfulness, relevance, depth, clarity, and the use of evidence-based information in their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
  "prompt_template": "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
}

judge_pair_prompt = judge_pair_json["system_prompt"] + "\n\n" + judge_pair_json["prompt_template"]