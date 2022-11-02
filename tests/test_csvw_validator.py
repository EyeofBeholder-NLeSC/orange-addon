from orangeext.csvw_validator import CSVWValidator
from orangewidget.tests.base import WidgetTest

# setup test environment
WidgetTest.setUpClass()


def test_validator_output():
    """
    This is to test if the output contains expected number of items.
    With the default setting of the widget, it should output a table
    with two rows that store URLs to two CSV files.
    """
    wt = WidgetTest()
    w = wt.create_widget(CSVWValidator)
    w.metadata_url = (
        "https://w3c.github.io/csvw/tests/test011/tree-ops.csv-metadata.json"
    )
    w.submit()
    output = wt.get_output("CSV URLs", widget=w)
    assert len(output) == 1
    assert output[0].metas[0].endswith("tree-ops.csv")
