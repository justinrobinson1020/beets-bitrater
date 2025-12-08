"""Tests that plugin display methods use logger instead of print."""

import ast
from pathlib import Path


class TestNoPrintInPlugin:
    """Verify plugin.py uses logger.info instead of print()."""

    def test_no_bare_print_calls(self) -> None:
        """plugin.py should not contain any print() function calls."""
        plugin_path = Path(__file__).parent.parent / "beetsplug" / "bitrater" / "plugin.py"
        source = plugin_path.read_text()
        tree = ast.parse(source)

        print_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for bare print() calls
                if isinstance(node.func, ast.Name) and node.func.id == "print":
                    print_calls.append(node.lineno)

        assert not print_calls, (
            f"Found print() calls at lines {print_calls}. " "Use logger.info() instead."
        )

    def test_no_emoji_characters(self) -> None:
        """plugin.py should use text markers [OK], [WARN], [FAIL] instead of emoji."""
        plugin_path = Path(__file__).parent.parent / "beetsplug" / "bitrater" / "plugin.py"
        source = plugin_path.read_text()

        emoji_chars = ["✓", "✗", "⚠️", "⚠"]
        found_emoji = []
        for i, line in enumerate(source.splitlines(), 1):
            for emoji in emoji_chars:
                if emoji in line:
                    found_emoji.append((i, emoji))

        assert not found_emoji, (
            f"Found emoji characters: {found_emoji}. "
            "Use [OK], [WARN], [FAIL] text markers instead."
        )
