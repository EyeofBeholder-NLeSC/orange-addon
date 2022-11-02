from orangeext.csvw_reader import CSVWReader
from orangewidget.tests.base import WidgetTest
from Orange.data.pandas_compat import table_from_frame

import pandas as pd

# setup test environment
WidgetTest.setUpClass()


def test_reader_output():
    wt = WidgetTest()
    w = wt.create_widget(CSVWReader)

    input_table = pd.DataFrame(
        data=["https://w3c.github.io/csvw/tests/test011/tree-ops.csv"],
        columns=["csv_url"],
    )
    print("input:", input_table)
    wt.send_signal(input="CSV URL", value=table_from_frame(input_table), widget=w)
    w.metadata_url = (
        "https://w3c.github.io/csvw/tests/test011/tree-ops.csv-metadata.json"
    )
    w.submit()
    output = wt.get_output("Data", widget=w)
    assert len(output) == 2
