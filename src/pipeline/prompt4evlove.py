complex_prompt = """I want you act as a Prompt Rewriter.
Your objective is to rewrite a given bio-medical prompt into a more complex version to make those famous AI systems(e.g., ChatGPT and GPT4) a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by human expert.

Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:. Also, please do not omit the input in #Given Prompt#.
You SHOULD complicate the given prompt using one of the following five methods:
[METHOD 1] Please add one more constraints/requirements into #Given Prompt#
[METHOD 2] Please replace general concepts with more specific concepts.
[METHOD 3] If #Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.
[METHOD 4] If #Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increased. 
[METHOD 5] Incorporate real-world data and case studies.Provide actual experimental data, suchas DNA sequences, gel images, or chromatograms, from published studies or ongoingresearch projects.

And you should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
'#Given Prompt#', '#Rewritten Prompt#', 'given prompt' and 'rewritten prompt' are not allowed to appear in.

#Rewritten Prompt# 
#Given Prompt#: 
{question}
#Rewritten Prompt#:"""

complex_prompt_mc = """Act as a Question Rewriter to make biomedical multiple-choice questions more challenging for AI systems like ChatGPT and GPT-4, while remaining reasonable for human experts to understand and answer.

Complicate the given question using one of these methods:

[METHOD 1] Add one more constraint or requirement.
[METHOD 2] Replace general concepts with more specific ones.
[METHOD 3] Make the choices hard to differentiate by adding more complex distractors.
[METHOD 4] If solvable with simple thinking, request multi-step reasoning.

Limit additions to 10-20 words. Ensure a unique answer exists among the choices.

Question:
{question}

Output JSON format:
```
{{
    "question": "Rewritten question in the format: 'xxx\n\nA. xxx\nB. xxx\nC. xxx\nD. xxx'",
    "answer": "A/B/C/D"
}}
```
"""

complex_prompt_mix = """I want you to act as a Biomedical Prompt Enhancer. Your objective is to take a given biomedical prompt and rewrite it into a more complex version that would be more challenging for advanced AI systems like ChatGPT and GPT-4 to handle, while still being reasonable and understandable to human experts.

## METHOD
To increase the complexity, you SHOULD use one or more of the following five methods, selecting the appropriate method(s) based on the characteristics of the original prompt:

[METHOD 1] Add one or more additional constraints or requirements to the prompt.

[METHOD 2] Replace general concepts with more specific or technical concepts.

[METHOD 3] If the original prompt can be solved with simple thinking processes, rewrite it to explicitly request multiple-step reasoning.

[METHOD 4] If the prompt contains inquiries about certain issues, increase the depth and breadth of the inquiry.

[METHOD 5] Incorporate real-world data and case studies, such as DNA sequences, gel images, or chromatograms from published studies or ongoing research projects.

You should aim to keep the rewritten prompt concise, adding only 10 to 20 words to the original prompt in each method. Avoid making the rewritten prompt verbose.

## Original Prompt
{question}

## Rewriting Process
[Method(s) used and changes made]

## Final Rewritten Prompt
"""