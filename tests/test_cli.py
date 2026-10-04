import argparse
import contextlib
import importlib
import io
import unittest

from numrev.cli import COMMANDS, Command, resolve


def every_command() -> list[tuple[str, Command]]:
    result = []
    for name, entry in COMMANDS.items():
        if isinstance(entry, Command):
            result.append((name, entry))
        else:
            result += [(f"{name} {sub}", command) for sub, command in entry.items()]
    return result


class CliTests(unittest.TestCase):
    def test_every_command_builds_its_parser(self):
        for name, command in every_command():
            with self.subTest(command=name):
                module = importlib.import_module(command.module)
                parser = argparse.ArgumentParser(prog=f"numrev {name}")
                module.add_arguments(parser)
                self.assertTrue(callable(module.run))

    def test_resolve(self):
        name, command, rest = resolve(["align", "nw", "a", "b", "--apply"])
        self.assertEqual((name, command.module, rest), ("align nw", "numrev.pipeline.align_nw", ["a", "b", "--apply"]))
        self.assertEqual(resolve(["tag", "x.entities.csv"])[0], "tag")
        self.assertEqual(resolve(["view", "alignment"])[2], ["alignment"])
        with contextlib.redirect_stderr(io.StringIO()):
            for argv in (["inconnue"], ["align"]):
                with self.assertRaises(SystemExit):
                    resolve(argv)


if __name__ == "__main__":
    unittest.main()
