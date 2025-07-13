import logging
import unittest

# External libraries
import polars as pl

from langchain_core.messages import HumanMessage

# Local modules
from src.agent._core import ChatState, RunTimeState
from src.agent._graph import build_graph
from src.agent.mock import MockLLM
from src.agent.tools import build_tools


class TestDataFrameTools(unittest.TestCase):

    def setUp(self) -> None:
        
        llm = MockLLM()

        # initialize runtime
        df = pl.DataFrame(
            {"col": [1, 2, 3, 4, 5]}
        )

        self.runtime = RunTimeState(
            df=df
        )

        _, get_df_property = build_tools(self.runtime)
        tools = [get_df_property]

        self.graph = build_graph(llm, tools)


    def test_pass(self):

        input_state = ChatState(
            messages=[
                HumanMessage(content="'Sup?")
            ]
        )

        output_state = self.graph.invoke(input_state)
        messages = output_state.get('messages', [])
        
        # Should have output messages
        self.assertGreater(len(messages), 0)

        last_message = messages[-1]
        logging.debug(f"{self.__class__.__name__} (test_pass): Last message - {last_message}")

        self.assertEqual(last_message.content, 'I have no idea what to do.')


    def test_runtime_change(self):

        input_state = ChatState(
            messages=[
                HumanMessage(content="What is the dataframe length?")
            ]
        )
    
        self.runtime["df"] = pl.DataFrame(
            {"col": [1, 2, 3, 4, 5, 6, 7, 8]}
        )

        output_state = self.graph.invoke(input_state)

        output_state = self.graph.invoke(input_state)
        messages = output_state.get('messages', [])
        
        # Should have output messages
        self.assertGreater(len(messages), 0)

        last_message = messages[-1]
        logging.debug(f"{self.__class__.__name__} (test_runtime_change): Last message - {last_message}")

        self.assertEqual(last_message.content, 'Tool output: len(df): 8')
        

if __name__ == "__main__":

        unittest.main()
    

    
    

    

    # input_state = ChatState(messages=[HumanMessage(content="Hey")])

    

    # print(state['messages'])