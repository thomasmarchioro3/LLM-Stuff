import logging

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolCall, ToolMessage

class MockLLM(Runnable):

    def invoke(
            self, 
            messages: list[BaseMessage], 
            config: RunnableConfig | None=None
        ):

        last_message = messages[-1] if messages else BaseMessage(content="")

        logging.debug(f"{self.__class__.__name__}: Type of last message received: {last_message.__class__.__name__}")

        curr_message = None
        if ("length" in last_message.content) and isinstance(last_message, HumanMessage):

            curr_message = AIMessage(content="Let's measure the df length.", tool_calls=[
                ToolCall(name="get_df_property", args={"prop": "len"}, id="1", type="tool_call"),
            ],
            )

        elif isinstance(last_message, ToolMessage):
            curr_message = AIMessage(content=f"Tool output: {last_message.content}")

        else:
            curr_message = AIMessage(content="I have no idea what to do.")
        
        return curr_message
    