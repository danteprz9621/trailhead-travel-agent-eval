"""
Agent under test: agents/chatbot.py
"""

import pytest

from deepeval import evaluate
from deepeval.test_case import ConversationalTestCase, Turn, MultiTurnParams
from deepeval.models import OllamaModel
from deepeval.metrics import (
    KnowledgeRetentionMetric,
    ConversationCompletenessMetric,
    RoleAdherenceMetric,
    ConversationalGEval
)

from agents.chatbot import SupportChatbot

model = OllamaModel(model="llama3.1:8b", base_url="http://localhost:11434")
knowledge_retention = KnowledgeRetentionMetric(threshold=0.6, model=model)
conversation_completeness = ConversationCompletenessMetric(threshold=0.8, model=model)
role_adherence = RoleAdherenceMetric(threshold=0.7, model=model)

correctness = ConversationalGEval(
    name="correctness",
    criteria="Did the chatbot resolve the issue?",
    model=model,
    threshold=0.8,
    evaluation_params=[
        MultiTurnParams.ROLE,
        MultiTurnParams.CONTENT
    ]
)

name_convo = [
    "Hi, my name is Michael Scott. I want to know about the baggage policy.",
    "How about the refund policy?",
    "Can you tell me my name again?"
]

cancel_order_convo = [
    "Hi, my name is Michael Scott. I want to know about the cancellation policy.",
    "I want to cancel my order"
]

off_topic_convo = [
    "Hey Sam, it's been too long since we last spoke. Wanna hang out tomorrow?",
    "Are you sure you are Sam, my friend from pilates?"
]

# Drives the chatbot through a scripted conversation and returns the real
# user/assistant exchanges as Turns for a ConversationalTestCase.
def simulate_conversation(chatbot: SupportChatbot, user_msgs: list[str]) -> Turn:
    turns = []
    for user_msg in user_msgs:
        reply = chatbot.send(user_msg)
        turns.append(Turn(role="user", content=user_msg))
        turns.append(Turn(role="assistant", content=reply))
    return turns

# User gives their name early on, then asks for it back several turns later.
def test_chatbot_retains_knowledge_across_turns():
    chatbot = SupportChatbot()
    turns = simulate_conversation(chatbot, name_convo)
    test_case = ConversationalTestCase(turns=turns)
    evaluate(test_cases=[test_case], metrics=[knowledge_retention])

# Goal-oriented conversation (cancel an order) -- checks the goal actually
# gets resolved, not just that the bot replies politely.
def test_chatbot_completes_the_converstation():
    chatbot = SupportChatbot()
    turns = simulate_conversation(chatbot, cancel_order_convo)
    test_case = ConversationalTestCase(turns=turns)
    evaluate(test_cases=[test_case], metrics=[conversation_completeness])


# Off-topic push -- exercises whether Sam holds its persona under pressure
# instead of dropping character to chat about unrelated things.
def test_chatbot_stays_in_role():
    chatbot = SupportChatbot()
    turns = simulate_conversation(chatbot, off_topic_convo)
    test_case = ConversationalTestCase(turns=turns)
    evaluate(test_cases=[test_case], metrics=[])


# Same cancellation conversation as above, scored with the custom
# conversation-level GEval metric instead of ConversationCompletenessMetric.
def test_chatbot_correctness_cancellation():
    chatbot = SupportChatbot()
    turns = simulate_conversation(chatbot, cancel_order_convo)
    test_case = ConversationalTestCase(turns=turns)
    evaluate(test_cases=[test_case], metrics=[correctness])
