from langchain.memory import ConversationBufferMemory
from datetime import datetime

class UserSession:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory = ConversationBufferMemory()
        self.last_activity = datetime.now()
        self.message_count = 0
        
    def update_activity(self):
        self.last_activity = datetime.now()
        self.message_count += 1
        
    def should_clear_history(self):
        # Clear history if:
        # 1. More than 10 messages in the conversation
        # 2. Or if the session is older than 30 minutes
        MAX_MESSAGES = 10
        MAX_SESSION_MINUTES = 30
        
        time_diff = (datetime.now() - self.last_activity).total_seconds() / 60
        return (self.message_count >= MAX_MESSAGES) or (time_diff >= MAX_SESSION_MINUTES)
    
    def clear_history(self):
        self.memory = ConversationBufferMemory()
        self.message_count = 0