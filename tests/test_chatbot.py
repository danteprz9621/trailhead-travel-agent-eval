"""
DeepEval test skeleton: Multi-turn Chatbot
Agent under test: agents/chatbot.py

DeepEval models multi-turn conversations with ConversationalTestCase + Turn.
Docs: https://docs.confident-ai.com/docs/multiturn-introduction
"""

import pytest
from deepeval import assert_test
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.metrics import (
    KnowledgeRetentionMetric,
    ConversationCompletenessMetric,
    RoleAdherenceMetric,
)

from agents.chatbot import SupportChatbot


# 1. Write a helper function simulate_conversation(chatbot, user_messages)
#    that drives `chatbot` through a scripted list of user messages and
#    returns a list of Turn objects (role="user" / role="assistant") built
#    from the REAL inputs/outputs -- not hardcoded strings


# 2. Write test_chatbot_retains_knowledge_across_turns():
#    - Instantiate SupportChatbot
#    - Script a conversation where the user gives a fact early on (e.g.
#      their name or order number) and asks the bot to recall it later
#    - Build the Turns using your helper from step 1
#    - Wrap them in a ConversationalTestCase
#    - Run assert_test() with a KnowledgeRetentionMetric


# 3. Write test_chatbot_completes_the_conversation():
#    - Script a conversation with a clear user goal
#      (e.g. "I want to cancel my order")
#    - Assert with ConversationCompletenessMetric that the goal got resolved


# 4. Write test_chatbot_stays_in_role():
#    - Try to get the chatbot to break character or go off-topic
#    - Pass a chatbot_role string describing Sam's intended persona into
#      your ConversationalTestCase
#    - Assert with RoleAdherenceMetric that it stayed on-role


# 5. (Stretch) Parametrize step 2 or 3 over multiple scripted conversations
#    using pytest.mark.parametrize
