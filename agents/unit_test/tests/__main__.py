import logging
import unittest

# Even though the LSP marks the imported tests as unused, they are run by unittest.main()
from .test_df_tools import TestDataFrameTools

if __name__ == "__main__":


    logging.basicConfig(
        # level=logging.INFO,
        level=logging.DEBUG,
        format="{levelname} - {message}",
        style="{",
        handlers=[
            logging.StreamHandler()
        ]
    )

    unittest.main()