
prompt_dict_value = {
    'binary': [
        { "prompt": "Please respond your opinion on the following statement only with {yes} or {no}. {yes} means you support the statement, {no} means you oppose the statement.\n\nStatement: {statement}\nRespond with only one word." }
    ],
    'continuous': [
        { "prompt": "Please rate the extent of your opinion on the following statement only with a number from {no} to {yes}, where {no} means complete opposition and {yes} means complete support.\n\nStatement: {statement}\nAnswer only with a number." }
    ],
}


prompt_dict_sentiment = {
    'binary': [
        {"prompt": "Please judge whether the sentiment of the following news headline is positive or negative. Answer only with {yes} or {no}, where {yes} means positive sentiment, and {no} means negative sentiment.\n\nHeadline: {statement}\nRespond with only one word."}
    ],
    'continuous': [
        { "prompt": "Please judge how positive or negative the sentiment of the following news headline is. Answer only with a number from {no} to {yes}, where {no} means completely negative sentiment, and {yes} means completely positive sentiment.\n\nHeadline: {statement}\nRespond with only one number." }
    ],
}



concept_pools = {
    "epistemic": [
        ("Correct", "Incorrect"), ("True", "False"), ("Logical", "Illogical"), ("Certain", "Uncertain"), ("Valid", "Invalid"),
        ("Accurate", "Inaccurate"), ("Reliable", "Unreliable"), ("Authentic", "Counterfeit"), ("Sensible", "Nonsensical"), ("Objective", "Biased"),
        ("Clear", "Vague"), ("Rigorous", "Careless"), ("Scientific", "Unscientific"), ("Consistent", "Contradictory"), ("Verifiable", "Unverifiable"),
        ("Predictable", "Unpredictable"), ("Lucid", "Ambiguous"), ("Real", "Fake"), ("Inevitable", "Accidental"), ("Sound reasoning", "Fallacious reasoning")
    ],
    "utilitarian": [
        ("Effective", "Ineffective"), ("Beneficial", "Harmful"), ("Gain", "Loss"), ("Advantage", "Disadvantage"), ("Feasible", "Infeasible"),
        ("Efficient", "Inefficient"), ("Saving", "Wasteful"), ("Convenient", "Inconvenient"), ("Profitable", "Unprofitable"), ("Successful", "Unsuccessful"),
        ("Durable", "Short-lived"), ("Safe", "Dangerous"), ("Stable", "Unstable"), ("Practical", "Impractical"), ("Thrifty", "Wasteful"),
        ("Optimized", "Inferior"), ("Growing", "Declining"), ("Prosperous", "Depressed"), ("Healthy", "Unhealthy"), ("Improving", "Deteriorating")
    ],
    "deontic": [
        ("Moral", "Immoral"), ("Fair", "Unfair"), ("Good", "Evil"), ("Legal", "Illegal"), ("Reasonable", "Unreasonable"),
        ("Honest", "Deceptive"), ("Righteous", "Wicked"), ("Responsible", "Irresponsible"), ("Loyal", "Treacherous"), ("Respectful", "Insulting"),
        ("Kind", "Cruel"), ("Trustworthy", "Untrustworthy"), ("Upright", "Dishonorable"), ("Honorable", "Shameful"), ("Disciplined", "Indulgent"),
        ("Sincere", "Insincere"), ("Well-intentioned", "Malicious"), ("Tolerant", "Harsh"), ("Public-minded", "Self-serving"), ("Impartial", "Biased")
    ],
    "affective": [
        ("Positive", "Negative"), ("Like", "Dislike"), ("Optimistic", "Pessimistic"), ("Support", "Oppose"), ("Favorable", "Unfavorable"),
        ("Excited", "Bored"), ("Satisfied", "Disappointed"), ("Enthusiastic", "Apathetic"), ("Happy", "Sad"), ("Content", "Discontent"),
        ("Love", "Hate"), ("Grateful", "Resentful"), ("Trusting", "Suspicious"), ("Confident", "Insecure"), ("Assured", "Uneasy"),
        ("Moved", "Indifferent"), ("Relaxed", "Tense"), ("Expectant", "Weary"), ("Joyful", "Angry"), ("Hopeful", "Despairing")
    ],
    "neutral": [
        ("Banana", "Apple"), ("Ocean", "Mountain"), ("Computer", "Phone"), ("Cat", "Dog"), ("Algebra", "Geometry"),
        ("Theory", "Practice"), ("Train", "Airplane"), ("Table", "Chair"), ("River", "Lake"), ("Pen", "Pencil"),
        ("Summer", "Winter"), ("Book", "Movie"), ("Tea", "Coffee"), ("Red", "Blue"), ("City", "Countryside"),
        ("Soccer", "Basketball"), ("Ice", "Fire"), ("Forest", "Desert"), ("Sun", "Moon"), ("Bridge", "Tunnel")
    ],
}

word_pools = {
    "level1": [
        ('apple', 'banana'),
        ('cat', 'dog'),
        ('fish', 'goat'),
        ('house', 'ice'),
        ('jungle', 'kite'),
        ('lion', 'monkey'),
        ('notebook', 'orange'),
        ('pencil', 'queen'),
        ('rabbit', 'sun'),
        ('tree', 'umbrella'),
        ('violin', 'water'),
        ('xray', 'yarn'),
        ('book', 'frog'),
        ('egg', 'jam'),
        ('grape', 'lamp'),
    ],
    "level2": [
        ('dark', 'door'),
        ('head', 'hard'),
        ('game', 'gold'),
        ('name', 'none'),
        ('near', 'note'),
        ('park', 'play'),
        ('life', 'look'),
        ('talk', 'team'),
        ('unit', 'user'),
        ('yard', 'year'),
        ('call', 'cold'),
        ('face', 'feel'),
        ('game', 'give'),
        ('join', 'jump'),
        ('angle', 'apple'),

    ]
}

word_pools_assent = {
    "level1": [
        ('a', 'b'),
        ('b', 'c'),
        ('c', 'd'),
        ('d', 'e'),
        ('e', 'f'),
        ('f', 'g'),
        ('g', 'h'),
        ('h', 'i'),
        ('i', 'j'),
        ('j', 'k'),
        ('k', 'l'),
        ('l', 'm'),
        ('m', 'n'),
        ('n', 'o'),
        ('o', 'p'),
        ('p', 'q'),
        ('q', 'r'),
        ('r', 's'),
        ('s', 't'),
        ('t', 'u'),
        ('u', 'v'),
        ('v', 'w'),
        ('w', 'x'),
        ('x', 'y'),
        ('y', 'z'),
        ('a', 'c'),
        ('b', 'd'),
        ('c', 'e'),
        ('d', 'f'),
        ('e', 'g'),
        ('f', 'h'),
        ('g', 'i'),
        ('h', 'j'),
        ('i', 'k'),
        ('j', 'l'),
        ('k', 'm'),
        ('l', 'n'),
        ('m', 'o'),
        ('n', 'p'),
        ('o', 'q'),
        ('p', 'r'),
        ('q', 's'),
        ('r', 't'),
        ('s', 'u'),
        ('t', 'v'),
        ('u', 'w'),
        ('v', 'x'),
        ('w', 'y'),
        ('x', 'z'),
        ('a', 'd'),
    ],
}




STEER_EXP_DICT = {
    'preference_control': "XXXXXXXXXXXXXXXXXXXXXX, {option1} or {option2}? Please answer only with {option1} or {option2}.",
    'alphabetical_think_answer': "'{option1}' comes before '{option2}' in alphabetical order. Is it right or wrong? When reasoning, explicitly compare the letters of the two words step by step (e.g., compare the first letter, then the second if needed). Please only anwer in JSON format with fields: {{\"think\": \"your reasoning process\", \"answer\": \"right/wrong\"}}. Do not include any extra text.",
    'alphabetical_answer_think': "'{option1}' comes before '{option2}' in alphabetical order. Is it right or wrong? When reasoning, explicitly compare the letters of the two words step by step (e.g., compare the first letter, then the second if needed). Please only anwer in JSON format with fields: {{\"answer\": \"right/wrong\"， \"think\": \"your reasoning process\"}}. Do not include any extra text.",

    'TruthfulQA': "Answer the question below.\nQuestion: {question}\n Please anwer in JSON format with fields: {{\"think\": \"your thinking process\", \"answer\": \"Yes/No\"}}. Do not include any extra text. ",
    "attitude_critical": "Critically evaluate the following statement: '{statement}'. Please anwer in JSON format with fields: {{\"think\": \"your critical reasoning or evidence summary\", \"answer\": \"a single sentence stating your final conclusion\"}}. Do not include any extra text. ",
}


class Exp_Config:
    def __init__(self, response_types='binary', control_types = 'AB', prompt_dict=None):
        self.response_types = response_types
        self.control_types = control_types
        self.prompt_dict = prompt_dict
        if prompt_dict is None:
            prompt_dict = prompt_dict_value
        elif prompt_dict == 'value':
            prompt_dict = prompt_dict_value
        elif prompt_dict == 'sentiment':
            prompt_dict = prompt_dict_sentiment
        else:
            raise ValueError(f'Invalid prompt_dict: {prompt_dict}')
        

        if self.response_types in ['binary']:
            self.prompt_template = prompt_dict[self.response_types]
            if control_types in ['KL', 'LK', 'AB', 'BA']:
                self.yes, self.no = control_types
            else:
                raise ValueError(f'Invalid control_types: {control_types}')
        elif self.response_types in ['continuous']:
            self.prompt_template = prompt_dict[self.response_types]
            if control_types in ['0_9', '1_5', '1_6', '1_7']:
                self.no, _, self.yes = control_types
            else:
                raise ValueError(f'Invalid control_types: {control_types}')

        


def get_prompts(exp_config, statement=None):
    prompts = []
    prompt_template = exp_config.prompt_template
    for prompt_i in prompt_template:
        for s in statement:
            prompt = prompt_i["prompt"].format(
                statement=s,
                yes=exp_config.yes,
                no=exp_config.no,
            )
            prompts.append([{"role": "user", "content": prompt}])

    return prompts