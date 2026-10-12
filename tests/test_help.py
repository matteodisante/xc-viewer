"""Info, How to use and Sources & methods: complete, linked, and never repetitive."""

import html
import re

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyproj")

from xc_viewer.thermal_imagery import LAYERS
from xc_viewer.thermal_ridges import LAYER
from xc_viewer.widgets.help_text import TOPICS, body, page
from xc_viewer.widgets.sources_methods import (
    SECTIONS,
    SourcesMethods,
    sources_html,
)


def _sentences(markup):
    """Normalised sentences of at least six words, without tags or entities."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", markup))
    parts = re.split(r"(?<=[.!?:;])\s+", " ".join(text.split()))
    return {part.lower().strip(" .:;") for part in parts if len(part.split()) >= 6}


def _all_help():
    return {
        (topic, kind): body(topic, kind)
        for topic in TOPICS
        for kind in ("info", "howto")
    }


def test_sources_page_names_every_data_and_map_source():
    text = sources_html()
    for name in ("FFVL", "Natural Earth", "RGE ALTI", LAYER, "Esri World Hillshade"):
        assert name in text
    for layer in ",".join(LAYERS.values()).split(","):
        assert layer in text
    assert "arxiv.org/abs/2601.01293" in text


def test_every_contents_link_has_its_section():
    text = sources_html()
    for anchor, _title in SECTIONS:
        assert f'href="#{anchor}"' in text
        assert f'name="{anchor}"' in text


def test_every_link_from_a_screen_opens_an_existing_section():
    anchors = {anchor for anchor, _title in SECTIONS}
    linked = set()
    for text in _all_help().values():
        linked |= set(re.findall(r'href="methods:([^"]+)"', text))
    assert linked
    assert linked <= anchors


def test_info_how_to_use_and_sources_never_repeat_a_sentence():
    roles = {"info": set(), "howto": set(), "methods": _sentences(sources_html())}
    for (_topic, kind), text in _all_help().items():
        roles[kind] |= _sentences(text)
    for first, second in (("info", "howto"), ("info", "methods"), ("howto", "methods")):
        assert not roles[first] & roles[second], (first, second)


def test_help_never_cites_people_by_first_name_or_the_thesis():
    texts = [sources_html()]
    texts += [page(topic, kind)[1] for topic in TOPICS for kind in ("info", "howto")]
    for text in texts:
        assert "Jérémie" not in text
        assert "thesis" not in text.lower()


def test_page_is_built_on_first_display(qapp):
    view = SourcesMethods()
    assert view._browser.toPlainText() == ""
    view.ensure_loaded()
    assert "Plane crossings" in view._browser.toPlainText()


@pytest.mark.parametrize("topic", list(TOPICS))
def test_each_screen_opens_and_reuses_its_info_and_how_to_use(qapp, topic):
    from PyQt6.QtWidgets import QDialog, QTextBrowser

    from xc_viewer.widgets.group_flights import GroupFlights
    from xc_viewer.widgets.map_view import MapView
    from xc_viewer.widgets.plot_controls import PlotControls
    from xc_viewer.widgets.route_comparison import RouteComparison
    from xc_viewer.widgets.route_locator import RouteLocator
    from xc_viewer.widgets.thermal_3d import Thermal3D
    from xc_viewer.widgets.thermal_density import ThermalDensity
    from xc_viewer.widgets.thermal_plane import ThermalPlane

    factories = {
        "trajectory": PlotControls,
        "map": MapView,
        "planes": ThermalPlane,
        "terrain": Thermal3D,
        "density": ThermalDensity,
        "routes": RouteComparison,
        "locator": lambda: RouteLocator(expanded=True),
        "groups": GroupFlights,
    }
    view = factories[topic]()
    try:
        view.show()
        qapp.processEvents()
        for button, kind, label in (
            (view._info, "info", "Info"),
            (view._howto, "howto", "How to use"),
        ):
            assert button.isVisible() and button.text() == label
            button.click()
            qapp.processEvents()
            window = button.window_
            assert window.isVisible() and not window.isModal()
            title, _html = page(topic, kind)
            assert window.windowTitle() == title
            assert window.findChild(QTextBrowser).toPlainText().startswith(title)
            window.close()
            button.click()
            qapp.processEvents()
            assert button.window_ is window and window.isVisible()
            window.close()
        assert len(view.findChildren(QDialog)) == 2
    finally:
        if hasattr(view, "shutdown"):
            view.shutdown()
        view.close()


def test_a_methods_link_opens_its_section_in_the_main_window(qapp, monkeypatch):
    from PyQt6.QtCore import QUrl

    from xc_viewer.main_window import MainWindow
    from xc_viewer.widgets.flight_picker import FlightPicker
    from xc_viewer.widgets.help import HelpButton
    from xc_viewer.widgets.info_browser import follow_link

    monkeypatch.setattr(FlightPicker, "_repopulate_filter_combos", lambda self: None)
    window = MainWindow()
    try:
        shown = []
        monkeypatch.setattr(
            window._sources_methods, "show_section", shown.append, raising=True
        )
        button = HelpButton("density", "info")
        button.click()
        follow_link(button.window_.browser, QUrl("methods:hours"))
        assert window._tabs.currentWidget() is window._sources_methods
        assert shown == ["hours"]
        button.window_.close()
    finally:
        window.close()
