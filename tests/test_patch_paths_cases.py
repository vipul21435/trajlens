"""Test-file edits the patch parser used to miss."""

from trajlens.metrics import is_test_path, new_files, patch_paths


def test_plain_diff_deleting_a_file():
    patch = "--- a/tests/test_x.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
    assert patch_paths(patch) == ["tests/test_x.py"]
    assert new_files(patch) == []


def test_plain_diff_with_timestamps():
    patch = "--- a/src/m.py\t2024-01-01 00:00:00\n+++ b/src/m.py\t2024-01-02 00:00:00\n"
    assert patch_paths(patch) == ["src/m.py"]


def test_rename_away_from_tests_keeps_both_paths():
    patch = (
        "diff --git a/tests/test_x.py b/src/x.py\n"
        "similarity index 90%\nrename from tests/test_x.py\nrename to src/x.py\n"
    )
    assert patch_paths(patch) == ["tests/test_x.py", "src/x.py"]


def test_quoted_and_spaced_paths():
    quoted = 'diff --git "a/tests/test a\\303\\251.py" "b/tests/test a\\303\\251.py"\n'
    assert patch_paths(quoted) == ["tests/test aé.py"]
    spaced = "diff --git a/tests/test a.py b/tests/test a.py\nnew file mode 100644\n"
    assert patch_paths(spaced) == ["tests/test a.py"]
    assert new_files(spaced) == ["tests/test a.py"]


def test_django_style_test_modules():
    assert is_test_path("django/contrib/auth/tests.py")
    assert is_test_path("app/test.py")
    assert not is_test_path("src/contest.py")
    assert not is_test_path("src/latest.py")


def test_hunk_lines_that_look_like_markers_are_not_paths():
    patch = "--- a/doc.txt\n+++ b/doc.txt\n@@ -1 +1 @@\n--- old rule\n+++ new rule\n"
    assert patch_paths(patch) == ["doc.txt"]
    two = patch + "--- a/b.txt\n+++ b/b.txt\n@@ -0,0 +1 @@\n+x\n"
    assert patch_paths(two) == ["doc.txt", "b.txt"]


def test_hunk_counts_decide_where_the_hunk_ends():
    # "-1" has no count (one line); the removed line starts with "-- ".
    patch = (
        "--- a/notes.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n--- removed rule\n"
        "--- a/gone.txt\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
        "--- a/src/b.py\n+++ b/src/b.py\n@@ -1,2 +0,0 @@\n--- one\n--- two\n"
        "--- a/c.txt\n+++ b/c.txt\n@@ -1 +1,2 @@\n ctx\n+++ added\n"
        "--- /dev/null\n+++ b/e.txt\n@@ -0,0 +1,2 @@\n+++ one\n+++ two\n"
        "--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1 @@\n ctx\n--- r1\n--- r2\n"
        "--- a/d.txt\n+++ b/d.txt\n"
    )
    want = ["notes.txt", "gone.txt", "src/b.py", "c.txt", "e.txt", "f.txt", "d.txt"]
    assert patch_paths(patch) == want


def test_unquoted_space_path_split_where_halves_agree():
    patch = "diff --git a/x b/y b/x b/y\nnew file mode 100644\n"
    assert patch_paths(patch) == ["x b/y"]
    assert new_files(patch) == ["x b/y"]
