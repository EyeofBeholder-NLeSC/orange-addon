from Orange.widgets.widget import OWBaseWidget, Input, Output
from Orange.widgets import gui
from Orange.data import Table
from Orange.data.pandas_compat import table_from_frame

from csvw import CSVW
import pandas as pd

from .utils import validate_metadata_url


class CSVWReader(OWBaseWidget):
    name = "CSVW Reader"
    description = "TBA"
    icon = "icons/csvw-reader.svg"

    class Inputs:
        csv_url = Input("CSV URL", Table, auto_summary=False)

    class Outputs:
        csv_data = Output("Data", Table, auto_summary=False)

    want_main_area = False

    def __init__(self):
        super().__init__()
        self.csv_url = ""
        self.metadata_url = ""

        gui.lineEdit(
            widget=self.controlArea,
            master=self,
            value="metadata_url",
            label="Metadata URL",
        )

        gui.button(
            widget=self.controlArea, master=self, label="Read", callback=self.submit
        )

    @Inputs.csv_url
    def set_csv_url(self, csv_url):
        # TODO: check row numbers here
        self.csv_url = csv_url[0].list[0]

    def submit(self):
        validate_metadata_url(self.metadata_url)
        result = CSVW(url=self.metadata_url, validate=True)
        if not result.is_valid:
            self.error("Validation Failed!")
            return

        for t in result.tables:
            base = t.base
            if t.url.resolve(base) == self.csv_url:
                output_table = table_from_frame(pd.DataFrame(t))
                self.Outputs.csv_data.send(output_table)
                return output_table

        self.error("Input invalid or not found in Metadata!")
