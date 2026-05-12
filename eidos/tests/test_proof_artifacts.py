from benchmarks.proof_artifacts import (
    EXPECTED_PROOF_ARTIFACT_FILENAMES,
    EXPECTED_PROOF_ARTIFACT_SUBDIRS,
    create_proof_artifact_dir,
    create_proof_subdirs,
    expected_proof_artifact_paths,
)


def test_create_proof_artifact_dir_creates_root_and_required_subdirs(tmp_path):
    out_dir = tmp_path / "proof_baseline_2026_05"

    result = create_proof_artifact_dir(out_dir)

    assert result == out_dir
    assert out_dir.is_dir()
    for subdir in EXPECTED_PROOF_ARTIFACT_SUBDIRS:
        assert (out_dir / subdir).is_dir()


def test_create_proof_subdirs_is_idempotent(tmp_path):
    out_dir = tmp_path / "proof_baseline_2026_05"

    first = create_proof_subdirs(out_dir)
    second = create_proof_subdirs(out_dir)

    assert first == second
    for path in second.values():
        assert path.is_dir()


def test_expected_proof_artifact_paths_lists_required_files_without_creating_them(tmp_path):
    out_dir = tmp_path / "proof_baseline_2026_05"

    paths = expected_proof_artifact_paths(out_dir)

    assert set(paths) == set(EXPECTED_PROOF_ARTIFACT_FILENAMES)
    assert paths["config.json"] == out_dir / "config.json"
    assert paths["run_manifest.json"] == out_dir / "run_manifest.json"
    for path in paths.values():
        assert not path.exists()

