from urllib import request
import json


def validate_metadata_url(url):
    assert url, "URL_EMPTY"

    try:
        url_file = request.urlopen(url)
    except ValueError:
        assert False, "URL_UNKNOWN_TYPE"

    try:
        data = json.load(url_file)
    except json.JSONDecodeError:
        assert False, "URL_INVALID"

    assert "@context" in data.keys(), "URL_NOT_JSONLD"
