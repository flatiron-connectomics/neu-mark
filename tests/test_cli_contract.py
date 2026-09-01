"""What the documentation build depends on, pinned here so it cannot quietly break.

The docs site renders the CLI reference from the real ``ArgumentParser``, which is what
stops published usage from drifting away from ``--help`` — so the parser has to be
reachable without running anything, and every subcommand has to be reachable from it. A
subcommand that is not is one that silently never gets documented.
"""

import argparse

import pytest


def test_build_parser_returns_the_parser_without_running_it():
    from neu_mark.cli import build_parser

    parser = build_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "neu-mark"


def test_every_subcommand_is_reachable_from_the_parser():
    from neu_mark import cli

    parser = cli.build_parser()
    subs = next(a for a in parser._actions
                if isinstance(a, argparse._SubParsersAction)).choices
    assert "help" in subs
    for name, sub in subs.items():
        assert sub.format_usage().strip(), f"{name} renders no usage line"


def test_help_is_a_subcommand_as_well_as_a_flag(capsys):
    """`neu-mark help points` and `neu-mark points --help` are the same thing to everyone
    except argparse, and being told "invalid choice" for one of them is a poor greeting
    from a tool whose whole surface is subcommands."""
    from neu_mark import cli

    assert cli.main(["help"]) == 0
    assert "usage: neu-mark" in capsys.readouterr().out

    assert cli.main(["help", "points"]) == 0
    printed = capsys.readouterr().out
    assert "usage: neu-mark points" in printed

    # the same text, because re-parsing means there is no second rendering to drift
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["points", "--help"])
    assert capsys.readouterr().out == printed


def test_help_for_something_that_is_not_a_command_lists_the_real_ones(capsys):
    from neu_mark import cli

    assert cli.main(["help", "nosuch"]) == 2, "argparse's own exit code for a bad choice"
    assert "invalid choice: 'nosuch'" in capsys.readouterr().err
