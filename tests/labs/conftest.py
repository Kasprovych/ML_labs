"""Фікстури тестів лабораторних: підготовлені «дані ЛР1» на синтетиці."""

import pytest

from mllabs.core.paths import ProjectPaths


@pytest.fixture(scope="session")
def lab1_project(tmp_path_factory, raw_kick_frame):
    """Тимчасовий проєкт, у якому ``data/processed/kick`` записано так, як це робить ЛР1.

    Замість повного запуску ЛР1 (довго) — ті самі кроки, що в його кроці 5:
    очищення, стратифіковане розбиття, запис train/test, схеми й маніфесту
    препроцесингу з типовими налаштуваннями.
    """
    from mllabs.data import (
        KICK_SCHEMA,
        ProcessedDatasetRepository,
        RawDataCleaner,
        StratifiedSplitter,
    )
    from mllabs.preprocessing import PreprocessorFactory

    project = ProjectPaths(root=tmp_path_factory.mktemp("lab2_project"))
    cleaning = RawDataCleaner(KICK_SCHEMA, max_missing_share=0.5).clean(raw_kick_frame)
    split = StratifiedSplitter(test_size=0.2, random_state=0).split(
        cleaning.data, cleaning.schema.target
    )
    repository = ProcessedDatasetRepository(project.data_processed_dir, cleaning.schema.name)
    repository.save(split, cleaning.schema, PreprocessorFactory(cleaning.schema, 0).manifest())
    return project
