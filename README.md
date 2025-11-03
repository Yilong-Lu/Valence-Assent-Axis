# Valence-Assent-Axis
Data for paper https://arxiv.org/abs/2510.27328

The `data` folder contains datasets used in the experiments.  
- The `value` subfolder holds data for the Value Judgment task.  
- The `sentiment` subfolder holds data for the Sentiment Analysis task.  
- `TruthfulQA_30.csv` contains 30 questions selected from the TruthfulQA dataset with yes/no answers, used for the Factual Judgment task.  
- `statement_general.csv` contains 30 LLM-generated statements on various topics, used for the Stance-Taking task.  

Prompts for each task and other experimental materials are located in the `scripts` folder, specifically in the `prompt.py` file.  

LLM responses for different reasoning tasks are provided in the `examples` folder.  

**Example:**  
Stance-Taking task of Qwen2.5-32B-Instruct:  
`examples/stance_taking/StanceTaking_qwen25_32b_results_extracted.json`  

```json
    {
        "layer": 43,
        "alpha": -97.4,
        "prompt_data": [
            {
                "role": "user",
                "content": "Critically evaluate the following statement: 'Students should be encouraged from using large language models for homework.'. Please anwer in JSON format with fields: {\"think\": \"your critical reasoning or evidence summary\", \"answer\": \"a single sentence stating your final conclusion\"}. Do not include any extra text. "
            }
        ],
        "responses": "\n{\"think\": \"The use of large language models for homework can potentially lead to academic dishonesty and hinder the development of critical thinking skills. However, it also presents opportunities for learning about technology's limitations and ethical use, and may serve as a tool for understanding complex topics, though this is often limited. Educational institutions and teachers need to consider these factors when deciding whether to allow such tools and how to integrate them into their curricula.\", \"answer\": \"Educational institutions must carefully consider the implications and establish clear guidelines; outright discouragement without exploration of educational value may not be beneficial.\"}",
        "statement": "Students should be encouraged from using large language models for homework.",
        "Group": "教育 (Education)",
        "CH": "应当鼓励学生使用大语言模型完成作业。",
        "think": "The use of large language models for homework can potentially lead to academic dishonesty and hinder the development of critical thinking skills. However, it also presents opportunities for learning about technology's limitations and ethical use, and may serve as a tool for understanding complex topics, though this is often limited. Educational institutions and teachers need to consider these factors when deciding whether to allow such tools and how to integrate them into their curricula.",
        "answer": "Educational institutions must carefully consider the implications and establish clear guidelines; outright discouragement without exploration of educational value may not be beneficial."
    },
```

**Field explanations:**
* `layer`: index of the intervention layer
* `alpha`: steering coefficient before normalization (ranges from negative to positive). 0 stands for baseline responses.
* `prompt_data`: content of the prompt
* `responses`: raw LLM output
* `think` and `answer`: extracted reasoning and final answer from the LLM