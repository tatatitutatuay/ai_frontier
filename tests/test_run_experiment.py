import unittest
from pathlib import Path

from ThaiSpoof.project.run_experiment import build_config_overrides, build_parser


class RunExperimentCliTest(unittest.TestCase):
    def test_parser_accepts_summary_stage_and_preset(self):
        args = build_parser().parse_args(
            [
                "--data-root",
                ".",
                "--preset",
                "smoke",
                "--stage",
                "summary",
            ]
        )

        self.assertEqual(args.data_root, Path("."))
        self.assertEqual(args.preset, "smoke")
        self.assertEqual(args.stage, "summary")

    def test_build_config_overrides_ignores_non_config_arguments_and_none_values(self):
        args = build_parser().parse_args(
            [
                "--data-root",
                ".",
                "--preset",
                "smoke",
                "--stage",
                "summary",
                "--batch-size",
                "12",
            ]
        )

        overrides = build_config_overrides(args)

        self.assertEqual(overrides, {"data_root": Path("."), "batch_size": 12})


if __name__ == "__main__":
    unittest.main()
