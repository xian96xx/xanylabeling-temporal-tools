from xal_temporal.integration import label_path_for_image


class _Widget:
    output_dir = None


def test_label_path_sibling_json():
    assert label_path_for_image(_Widget(), "/data/seq/00001.jpg") == "/data/seq/00001.json"


def test_label_path_output_dir():
    widget = _Widget()
    widget.output_dir = "/labels"
    assert label_path_for_image(widget, "/data/seq/00001.jpeg") == "/labels/00001.json"
