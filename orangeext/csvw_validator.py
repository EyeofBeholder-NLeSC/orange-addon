from Orange.widgets.widget import OWBaseWidget, Output
from Orange.widgets import gui
from Orange.data import Table, Domain, StringVariable

from csvw import CSVW

from orangeext.utils import validate_metadata_url


class CSVWValidator(OWBaseWidget):
    name = "CSVW Validator"
    description = "TBA"
    icon = "icons/csvw-validator.svg"

    class Outputs:
        csv_urls = Output("CSV URLs", Table, auto_summary=False)

    want_main_area = False

    def __init__(self):
        super().__init__()
        self.metadata_url = "https://raw.githubusercontent.com/EyeofBeholder-NLeSC/assessments-ontology/fix-metadata/metadata.json"

        gui.lineEdit(
            widget=self.controlArea,
            master=self,
            value="metadata_url",
            label="Metadata URL",
        )

        gui.button(
            widget=self.controlArea, master=self, label="Run", callback=self.submit
        )

    def submit(self):
        # validate url
        try:
            validate_metadata_url(self.metadata_url)
        except AssertionError as e:
            match str(e):
                case "URL_EMPTY":
                    err_msg = "Metadata URL empty!"
                case "URL_UNKNOWN_TYPE":
                    err_msg = "Metadata URL unknown!"
                case "URL_INVALID":
                    err_msg = "Metadata file not in JSON format!"
                case "URL_NOT_JSONLD":
                    err_msg = "Metadata file not in JSON-LD format!"
            self.error(err_msg)
            return

        # validate csv
        result = CSVW(url=self.metadata_url, validate=True)
        if not result.is_valid:
            self.error("Validation Failed!")
            return

        # output csv urls
        csv_urls = []
        for i in range(len(result.tables)):
            base = result.tables[i].base
            csv_urls.append([result.tables[i].url.resolve(base)])

        output_table = Table(
            Domain([], metas=[StringVariable(name="csv_url")]), csv_urls
        )
        self.Outputs.csv_urls.send(output_table)


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(CSVWValidator).run()
