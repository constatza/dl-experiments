"""Tests for checkpoint path derivation functions.

Tests the path derivation logic that auto-derives checkpoint paths from
experiment configuration following the convention:
{output_root}/{data_config_stem}/{model_name}/checkpoints/{model_name}.ckpt
"""

from __future__ import annotations

from pathlib import Path


from neuralls.platform.storage.arrays import derive_checkpoint_path
from neuralls.platform.storage.filesystem import extract_model_name


class TestExtractModelName:
    """Tests for extract_model_name function."""

    def test_extract_from_experiment_name(self, tmp_path: Path) -> None:
        """Test extracting model name from lower-case experiment.name."""
        config_path = tmp_path / "test_model.toml"
        config_path.write_text("""
[run]
type = "train"
seed = 42

[experiment]
name = "my_model"
""")

        model_name = extract_model_name(config_path)
        assert model_name == "my_model"

    def test_extract_from_inline_model_name_when_experiment_missing(self, tmp_path: Path) -> None:
        """Test using lower-case model.name when experiment.name is missing."""
        config_path = tmp_path / "linear.toml"
        config_path.write_text("""
[model]
name = "SomeModel"
""")

        model_name = extract_model_name(config_path)
        assert model_name == "SomeModel"

    def test_extract_from_referenced_model_profile(self, tmp_path: Path) -> None:
        """Test using a referenced lower-case model profile when job is otherwise thin."""
        profile_path = tmp_path / "model-profile.toml"
        profile_path.write_text("""
[model]
name = "ProfileModel"
""")
        config_path = tmp_path / "linear.toml"
        config_path.write_text(
            f"""
[run]
type = "train"
seed = 42
model = "{profile_path.name}"
"""
        )

        model_name = extract_model_name(config_path)
        assert model_name == "ProfileModel"

    def test_extract_falls_back_to_filename_when_identity_missing(self, tmp_path: Path) -> None:
        """Test fallback to filename stem when no lower-case identity fields exist."""
        config_path = tmp_path / "gnn.toml"
        config_path.write_text("""
[run]
type = "train"
seed = 42
""")

        model_name = extract_model_name(config_path)
        assert model_name == "gnn"

    def test_extract_from_filename_when_file_unreadable(self, tmp_path: Path) -> None:
        """Test fallback to filename when config file can't be read."""
        config_path = tmp_path / "nonexistent.toml"
        # File doesn't exist - should fallback to stem
        model_name = extract_model_name(config_path)
        assert model_name == "nonexistent"


class TestDeriveCheckpointPath:
    """Tests for derive_checkpoint_path function."""

    def test_basic_path_derivation(self, tmp_path: Path) -> None:
        """Test basic checkpoint path derivation."""
        model_config = tmp_path / "configs" / "linear.toml"
        model_config.parent.mkdir(parents=True)
        model_config.write_text("""
[run]
type = "train"
seed = 42

[experiment]
name = "ffnn"
""")

        data_config = tmp_path / "data-configs" / "test-solutions.toml"
        output_root = tmp_path / "output"

        checkpoint_path = derive_checkpoint_path(model_config, data_config, output_root)

        expected = output_root / "test-solutions" / "ffnn" / "checkpoints" / "ffnn.ckpt"
        assert checkpoint_path == expected

    def test_path_derivation_with_different_stems(self, tmp_path: Path) -> None:
        """Test path derivation with different config names."""
        model_config = tmp_path / "configs" / "linear.toml"
        model_config.parent.mkdir(parents=True)
        model_config.write_text("""
[run]
type = "train"
seed = 42

[experiment]
name = "linear"
""")

        data_config = tmp_path / "data-configs" / "collect-504.toml"
        output_root = tmp_path / "output"

        checkpoint_path = derive_checkpoint_path(model_config, data_config, output_root)

        expected = output_root / "collect-504" / "linear" / "checkpoints" / "linear.ckpt"
        assert checkpoint_path == expected

    def test_path_derivation_uses_experiment_name_not_filename(self, tmp_path: Path) -> None:
        """Test that lower-case experiment.name is used, not config filename."""
        model_config = tmp_path / "configs" / "my_config_file.toml"
        model_config.parent.mkdir(parents=True)
        model_config.write_text("""
[run]
type = "train"
seed = 42

[experiment]
name = "custom_model_name"
""")

        data_config = tmp_path / "data-configs" / "test.toml"
        output_root = tmp_path / "output"

        checkpoint_path = derive_checkpoint_path(model_config, data_config, output_root)

        # Should use experiment.name, not filename
        expected = (
            output_root / "test" / "custom_model_name" / "checkpoints" / "custom_model_name.ckpt"
        )
        assert checkpoint_path == expected

    def test_path_derivation_with_absolute_paths(self, tmp_path: Path) -> None:
        """Test path derivation with absolute paths."""
        model_config = tmp_path / "configs" / "linear.toml"
        model_config.parent.mkdir(parents=True)
        model_config.write_text("""
[run]
type = "train"
seed = 42

[experiment]
name = "ffnn"
""")

        data_config = tmp_path / "data-configs" / "test-eigenvector.toml"
        output_root = tmp_path / "output"

        checkpoint_path = derive_checkpoint_path(model_config, data_config, output_root)

        expected = output_root / "test-eigenvector" / "ffnn" / "checkpoints" / "ffnn.ckpt"
        assert checkpoint_path == expected

    def test_path_derivation_with_string_paths(self, tmp_path: Path) -> None:
        """Test that function accepts string paths."""
        model_config = tmp_path / "configs" / "gnn.toml"
        model_config.parent.mkdir(parents=True)
        model_config.write_text("""
[run]
type = "train"
seed = 42

[experiment]
name = "gnn"
""")

        data_config_str = str(tmp_path / "data-configs" / "test.toml")
        output_root_str = str(tmp_path / "output")

        checkpoint_path = derive_checkpoint_path(
            str(model_config), data_config_str, output_root_str
        )

        expected = tmp_path / "output" / "test" / "gnn" / "checkpoints" / "gnn.ckpt"
        assert checkpoint_path == expected


class TestPathConventionConsistency:
    """Tests ensuring path convention is consistent across codebase."""

    def test_matches_workflow_convention(self, tmp_path: Path) -> None:
        """Test that derived path matches prepare_experiment_outputs convention."""
        model_config = tmp_path / "linear.toml"
        model_config.write_text("""
[run]
type = "train"
seed = 42

[experiment]
name = "test_model"
""")

        data_config = tmp_path / "data.toml"
        output_root = tmp_path / "output"

        checkpoint_path = derive_checkpoint_path(model_config, data_config, output_root)

        # Should match: {output_root}/{data_config_stem}/{model_name}/checkpoints/{model_name}.ckpt
        assert checkpoint_path.parts[-5] == "output"  # output_root
        assert checkpoint_path.parts[-4] == "data"  # data_config_stem
        assert checkpoint_path.parts[-3] == "test_model"  # model_name
        assert checkpoint_path.parts[-2] == "checkpoints"  # checkpoints dir
        assert checkpoint_path.parts[-1] == "test_model.ckpt"  # checkpoint filename

    def test_convention_examples(self, tmp_path: Path) -> None:
        """Test real-world examples from experiments.toml."""
        examples = [
            {
                "model_name": "ffnn",
                "data_stem": "test-solutions",
                "expected_suffix": "test-solutions/ffnn/checkpoints/ffnn.ckpt",
            },
            {
                "model_name": "linear",
                "data_stem": "test-eigenvector-solutions-largest",
                "expected_suffix": "test-eigenvector-solutions-largest/linear/checkpoints/linear.ckpt",
            },
            {
                "model_name": "gnn",
                "data_stem": "collect-2040",
                "expected_suffix": "collect-2040/gnn/checkpoints/gnn.ckpt",
            },
        ]

        output_root = tmp_path / "output"

        for example in examples:
            model_config = tmp_path / f"{example['model_name']}.toml"
            model_config.write_text(f"""
[run]
type = "train"
seed = 42

[experiment]
name = "{example["model_name"]}"
""")

            data_config = tmp_path / f"{example['data_stem']}.toml"

            checkpoint_path = derive_checkpoint_path(model_config, data_config, output_root)

            expected = output_root / example["expected_suffix"]
            assert checkpoint_path == expected
