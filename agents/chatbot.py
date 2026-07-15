"""A minimal multi-turn chatbot that keeps conversation history in memory."""

import os

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DEFAULT_ROLE = (
    "You are 'Sam', a support chatbot for Trailhead Travel. You help "
    "customers manage bookings (change, cancel, look up order details). "
    "Stay in character as Sam at all times and never discuss anything "
    "unrelated to travel bookings."
)


class SupportChatbot:
    """Stateful chatbot: call .send() repeatedly to hold a conversation."""

    def __init__(self, chatbot_role: str = DEFAULT_ROLE, model: str = "gpt-4o-mini"):
        self.chatbot_role = chatbot_role
        self.model = model
        self.history = [{"role": "system", "content": chatbot_role}]

    def send(self, user_message: str) -> str:
        """Send one user message, append it to history, and return the reply."""
        self.history.append({"role": "user", "content": user_message})
        response = client.chat.completions.create(
            model=self.model,
            messages=self.history,
        )
        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        """Clear the conversation, keeping only the system role."""
        self.history = [{"role": "system", "content": self.chatbot_role}]


if __name__ == "__main__":
    bot = SupportChatbot()
    print(bot.send("Hi, my name is Dante and my order number is TT-4821."))
    print(bot.send("Can you remind me what my order number was?"))
