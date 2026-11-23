import importlib.util
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

from aspect_detection.visualization_data import load_visualization_data

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "runs" / "autodl-20261123" / "runs"


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib is an optional figure dependency")
class TestSingleFigureExports(unittest.TestCase):
    def test_each_visual_is_exported_to_its_own_file(self):
        from reports.plot_evaluation import export_figures

        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temporary:
            path = Path(temporary)
            export_figures(load_visualization_data(RESULTS), path, formats=("svg",))
            expected = {
                "fig01_aspect_f1": "口味",
                "fig02_overall_f1": "端到端负面",
                "fig03_confusion_baseline": "传统基线",
                "fig04_confusion_bert": "联合 BERT",
                "fig05_evidence_candidates": "无候选",
                "fig06_evidence_example": "等了四十分钟才上菜",
            }
            self.assertEqual({file.stem for file in path.glob("*.svg")}, set(expected))
            for stem, label in expected.items():
                with self.subTest(figure=stem):
                    tree = ElementTree.parse(path / f"{stem}.svg")
                    ids = {node.get("id") for node in tree.iter() if node.get("id")}
                    axes = {identifier for identifier in ids if identifier.startswith("axes_")}
                    self.assertEqual(len(axes), 2 if "confusion" in stem else 1)
                    self.assertIn(label, "".join(tree.getroot().itertext()))
                    original_heights_mm = [158, 87, 102, 102, 81, 51]
                    nominal_points = original_heights_mm[int(stem[3:5]) - 1] * 72 / 25.4
                    export_height = float(tree.getroot().get("viewBox").split()[3])
                    self.assertLess(export_height, nominal_points - 4)

    def test_comparisons_have_independent_palettes(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex
        from reports.plot_evaluation import aspect_figure, configure_style, confusion_figure, overall_figure

        configure_style()
        data = load_visualization_data(RESULTS)
        aspect = aspect_figure(data)
        overall = overall_figure(data)
        baseline = confusion_figure(data, 0)
        bert = confusion_figure(data, 1)
        for figure in (aspect, overall, baseline, bert):
            self.addCleanup(plt.close, figure)

        def bar_colors(figure):
            return {to_hex(bar.get_edgecolor()) for bar in figure.axes[0].patches}

        self.assertTrue(bar_colors(aspect).isdisjoint(bar_colors(overall)))
        baseline_map = baseline.axes[0].images[0].get_cmap()
        bert_map = bert.axes[0].images[0].get_cmap()
        self.assertNotEqual(to_hex(baseline_map(1.0)), to_hex(bert_map(1.0)))
        self.assertEqual(baseline.axes[0].images[0].get_clim(), bert.axes[0].images[0].get_clim())
        self.assertEqual(to_hex(baseline_map(0.0)), "#fdf390")
        self.assertEqual(to_hex(baseline_map(1.0)), "#9e7fa9")

    def test_aspect_headers_precede_indented_labels(self):
        import matplotlib.pyplot as plt
        from reports.plot_evaluation import aspect_figure, configure_style

        configure_style()
        data = load_visualization_data(RESULTS)
        figure = aspect_figure(data)
        self.addCleanup(plt.close, figure)
        positions = {text.get_text(): text.get_position() for text in figure.axes[0].texts}
        for row in data["aspects"]:
            header_x, header_y = positions[row["group"]]
            child_x, child_y = positions[row["name"]]
            self.assertLess(header_x, child_x)
            self.assertLess(header_y, child_y)
        self.assertEqual(len({positions[row["name"]][0] for row in data["aspects"]}), 1)

