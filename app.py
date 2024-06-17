import os
import streamlit as st
import textwrap
from pprint import pprint
from langchain import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import pandas as pd

# Define response schemas and judging criteria
response_schema_propositions = [
    ResponseSchema(name="Proposition 1", description="The First Proposition based on the Topic."),
    ResponseSchema(name="Proposition 2", description="The Second Proposition based on the Topic."),
    ResponseSchema(name="Proposition 3", description="The Third Proposition based on the Topic.")
]

response_schema_score = [
    ResponseSchema(name="Participant Name", description="The participant's name"),
    ResponseSchema(name="For or Against", description="Whether argument was for or against"),
    ResponseSchema(name="Score for Organization & Clarity (out of 20)", description="The judge's score for the organization & clarity section of the assessment"),
    ResponseSchema(name="Organization & Clarity Score Details", description="Details on the score for the organization & clarity section was determined"),
    ResponseSchema(name="Score for Strategy & Style (out of 40)", description="The judge's score for the strategy & style section of the assessment"),
    ResponseSchema(name="Strategy & Style Score Details", description="Details on how the score for the strategy & style section was determined"),
    ResponseSchema(name="Score for Effectiveness of Argument, Evidence and Content (out of 40)", description="The judge's score for the effectiveness of argument, evidence and content section of the assessment"),
    ResponseSchema(name="Effectiveness of Argument, Evidence and Content Score Details", description="Details on how the score for the effectiveness of argument, evidence and content section was determined"),
    ResponseSchema(name="Overall Score (out of 100)", description="The judge's overall score for the assessment of the participant's argument"),
    ResponseSchema(name="Overall Assessment Label", description="The one word label that corresponds to the total score"),
    ResponseSchema(name="Overall Assessment Summary", description="A 2 sentence overall assessment of the participant's argument")
]

judging_criteria = """This is the debate judge methodology for evaluating the argument and assigning a score out of a possible total 100 points:
Section One - Organization and Clarity - Maximum Score of 20 points assigned as follows:
0-5 points: Poorly organized, unclear structure, and difficult to follow.
6-10 points: Somewhat organized, somewhat clear structure, and moderately easy to follow.
11-15 points: Well-organized, clear structure, and easy to follow.
16-20 points: Exceptionally organized, very clear structure, and extremely easy to follow.
Section Two - Strategy & Style - Maximum Score of 40 points assigned as follows:
0-10 points: Poor strategy and style, lacks engagement, extremely underwhelming and ineffective use of rhetoric and language.
11-20 points: Fair strategy and style, somewhat engaging, underwhelming and limited use of rhetoric and language.
21-30 points: Good strategy and style, engaging, convincing, and effective use of rhetoric and language.
31-40 points: Excellent strategy and style, highly engaging, compellingly convincing, and masterful use of rhetoric and language.
Section Three - Effectiveness of Argument, Evidence, and Content - Maximum Score of 40 points assigned as follows
0-10 points: Weak argument with major logical flaws, false evidence or content, or lack of supporting evidence.
11-20 points: Somewhat effective argument with some logical flaws, limited false evidence or content, or insufficient supporting evidence.
21-30 points: Effective argument with minor logical flaws, minimal false evidence or content, or sufficient supporting evidence.
31-40 points: Highly effective argument with impeccable logic, no false evidence or content, and compelling supporting evidence.
"""

# Define prompt templates
moderator_prompt_txt = """Act as a seasoned debate moderator who is coordinating a debate among two or more parties. Based on the topic '{topic}', you will formulate 3 propositions that could be debated by the parties. An example proposition for climate change would be "Climate change is substantially caused by humans". Alternatively, you could state it as a question, "Is climate change substantially caused by humans?\n{proposition_format_instructions}"""

participant_prompt_txt = """Act as the participant in a multi-party debate. Your name is '{participant}.' You have been presented with the following proposition; '{proposition}' Your goal is to win the debate. To accomplish your goal, you will decide whether to make an argument for or against the proposition. Once you have decided to make an argument for or against the proposition, you will make the best possible argument. The argument will be evaluated and scored by an independent expert debate judge based on Organization and Clarity (20% of your score), Strategy & Style (40% of your score), and Effectiveness of Argument, Evidence, and Content (40% of your score). Do not provide both an argument for and an argument against. Take your time and focus on presenting your argument in a compelling, eloquent, and convincing manner to score as many points as possible and win the debate. Your argument must be no more than 300 words."""

judge_prompt_txt = """Act as a seasoned debate judge that is being asked to evaluate and score the arguments made by debate participants on the following proposition: '{proposition}'. {participant} made the following argument:'{participant_argument}' For the argument in question, you will evaluate the argument, conduct an assessment, and assign a score based on the following criteria and methodology:{judging_criteria} Here is a table that summarizes how overall scores correspond to the overall quality label for the argument: 81-100 points: Excellent 61-80 points: Good 36-60 points: Fair 0-35: Poor These labels should be used in providing the 2 sentence overall assessment.\n{score_format_instructions}"""

# Define classes
class DebateModerator:
    def __init__(self, model_name, prompt_template_text, response_schema, api_key=None, base_url=None):
        if base_url and api_key:
            self.model = ChatOpenAI(base_url=base_url, api_key=api_key, model=model_name)
        else:
            self.model = ChatOpenAI(model=model_name)
        output_parser = StructuredOutputParser.from_response_schemas(response_schema)
        prompt_template = PromptTemplate(
            template=prompt_template_text,
            input_variables=["topic"],
            partial_variables={"proposition_format_instructions": output_parser.get_format_instructions()}
        )
        self.chain = prompt_template | self.model | output_parser

    def generate_propositions(self, topic):
        return self.chain.invoke({"topic": topic})

class DebateParticipant:
    def __init__(self, name, model_name, prompt_template_text, api_key=None, base_url=None):
        self.name = name
        if base_url and api_key:
            self.model = ChatOpenAI(base_url=base_url, api_key=api_key, model=model_name)
        else:
            self.model = ChatOpenAI(model=model_name)
        prompt_template = PromptTemplate(
            input_variables=["proposition", "participant"],
            template=prompt_template_text,
        )
        self.chain = prompt_template | self.model

    def present_argument(self, proposition):
        return self.chain.invoke({"proposition": proposition, "participant": self.name})

class DebateJudge:
    def __init__(self, model_name, prompt_template_text, response_schema, api_key=None, base_url=None):
        if base_url and api_key:
            self.model = ChatOpenAI(base_url=base_url, api_key=api_key, model=model_name)
        else:
            self.model = ChatOpenAI(model=model_name)
        output_parser = StructuredOutputParser.from_response_schemas(response_schema)
        prompt_template = ChatPromptTemplate(
            input_variables=["proposition", "participant", "participant_argument", "judging_criteria"],
            messages=[HumanMessagePromptTemplate.from_template(prompt_template_text)],
            partial_variables={"score_format_instructions": output_parser.get_format_instructions()}
        )
        self.chain = prompt_template | self.model | output_parser

    def evaluate_argument(self, proposition, participant, argument):
        return self.chain.invoke({
            "proposition": proposition,
            "participant": participant,
            "participant_argument": argument,
            "judging_criteria": judging_criteria
        })

    def print_evaluation(self, evaluation):
        key_order = [
            'Participant Name', 'Overall Score (out of 100)', 'Overall Assessment Label',
            'For or Against', 'Overall Assessment Summary', 'Score for Organization & Clarity (out of 20)',
            'Organization & Clarity Score Details', 'Score for Strategy & Style (out of 40)',
            'Strategy & Style Score Details', 'Score for Effectiveness of Argument, Evidence and Content (out of 40)',
            'Effectiveness of Argument, Evidence and Content Score Details'
        ]
        evaluation_clean = {key: evaluation[key] for key in key_order}
        for key, value in evaluation_clean.items():
            print(f"{key} : {textwrap.fill(value, 100)}")

# Streamlit UI
st.title("LLM Simulated Debate")

openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
together_api_key = st.text_input("Enter your TogetherAI API Key:", type="password")

os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ['TOGETHER_API_KEY'] = together_api_key

model_options = [
    {"name": "GPT-3.5-turbo", "model_name": "gpt-3.5-turbo"},
    {"name": "GPT-4o", "model_name": "gpt-4o"},
    {"name": "Llama3-70B", "model_name": "meta-llama/Llama-3-70b-chat-hf", "api_key": together_api_key, "base_url": "https://api.together.xyz/v1"},
    {"name": "Mixtral 8x7B instruct", "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1", "api_key": together_api_key, "base_url": "https://api.together.xyz/v1"},
    {"name": "Llama3-8B", "model_name": "meta-llama/Llama-3-8b-chat-hf", "api_key": together_api_key, "base_url": "https://api.together.xyz/v1"}
]

num_participants = st.number_input("Select number of participants:", min_value=1, max_value=3, value=2)
num_judges = st.number_input("Select number of judges:", min_value=1, max_value=3, value=1)

moderator_model = st.selectbox("Select the model for Moderator:", model_options, format_func=lambda x: x['name'])
participant_models = [st.selectbox(f"Select the model for Participant {i+1}:", model_options, format_func=lambda x: x['name']) for i in range(num_participants)]
judge_models = [st.selectbox(f"Select the model for Judge {i+1}:", model_options, format_func=lambda x: x['name']) for i in range(num_judges)]

topic = st.text_input("Enter the debate topic:")

if st.button("Generate Propositions"):
    try:
        moderator = DebateModerator(
            model_name=moderator_model['model_name'],
            prompt_template_text=moderator_prompt_txt,
            response_schema=response_schema_propositions,
            api_key=moderator_model.get('api_key'),
            base_url=moderator_model.get('base_url')
        )

        st.write(f"Topic: {topic}")
        moderator_topics = moderator.generate_propositions(topic)
        st.session_state['moderator_topics'] = moderator_topics  # Store the generated propositions in session state
        st.session_state['proposition_selected'] = False

    except Exception as e:
        st.error(f"An error occurred while generating propositions: {e}")

# Check if propositions are already generated and stored in session state
if 'moderator_topics' in st.session_state and st.session_state['moderator_topics']:
    st.write("Generated Propositions:")
    st.write(st.session_state['moderator_topics'])
    selected_proposition = st.selectbox("Select Proposition:", [f"Proposition {i+1}" for i in range(3)])

    if selected_proposition and st.button("Confirm Proposition Selection"):
        st.session_state['selected_proposition'] = selected_proposition
        st.session_state['proposition_selected'] = True
        st.write(f"Selected proposition: {selected_proposition}")

if st.session_state.get('proposition_selected', False) and st.button("Run Debate"):
    try:
        proposition_key = st.session_state['selected_proposition']
        proposition = st.session_state['moderator_topics'][proposition_key]

        if proposition:
            st.write(f"The proposition to be argued is: {proposition}")

            participants = [
                DebateParticipant(
                    name=f"Participant {i+1} ({participant_models[i]['name']})",
                    model_name=participant_models[i]['model_name'],
                    prompt_template_text=participant_prompt_txt,
                    api_key=participant_models[i].get('api_key'),
                    base_url=participant_models[i].get('base_url')
                ) for i in range(num_participants)
            ]

            judges = [
                DebateJudge(
                    model_name=judge_models[i]['model_name'],
                    prompt_template_text=judge_prompt_txt,
                    response_schema=response_schema_score,
                    api_key=judge_models[i].get('api_key'),
                    base_url=judge_models[i].get('base_url')
                ) for i in range(num_judges)
            ]

            results = []

            for participant in participants:
                participant_argument = participant.present_argument(proposition)
                st.write(f"{participant.name}'s argument:")
                st.write(textwrap.fill(participant_argument.content, 100))

                for judge in judges:
                    judge_evaluation = judge.evaluate_argument(proposition, participant.name, participant_argument.content)
                    result = {
                        "Participant": participant.name,
                        "Judge": judge.model.model_name,
                        "Organization & Clarity": judge_evaluation['Score for Organization & Clarity (out of 20)'],
                        "Strategy & Style": judge_evaluation['Score for Strategy & Style (out of 40)'],
                        "Effectiveness of Argument, Evidence and Content": judge_evaluation['Score for Effectiveness of Argument, Evidence and Content (out of 40)'],
                        "Overall Score": judge_evaluation['Overall Score (out of 100)'],
                        "Evaluation Details": judge_evaluation
                    }
                    results.append(result)

            # Convert results to a DataFrame and display it as a table
            results_df = pd.DataFrame(results, columns=["Participant", "Judge", "Organization & Clarity", "Strategy & Style", "Effectiveness of Argument, Evidence and Content", "Overall Score"])
            st.write(f"Proposition: {proposition}")
            st.write("Debate Results:")
            st.table(results_df)

            # Display detailed evaluations in expanders
            for result in results:
                with st.expander(f"Evaluation Details for {result['Participant']} by {result['Judge']}"):
                    st.json(result['Evaluation Details'])

        else:
            st.warning("Please select a proposition before running the debate.")
    except Exception as e:
        st.error(f"An error occurred while running the debate: {e}")
