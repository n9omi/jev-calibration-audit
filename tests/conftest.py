from pathlib import Path

import pytest

from jev_audit.questions import QuestionBank

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def bank() -> QuestionBank:
    return QuestionBank.from_yaml(REPO / "configs" / "questions.yaml")
