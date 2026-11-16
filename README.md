# ML_labs

Інфраструктура для лабораторних робіт з моделювання ШІ: Python 3.12 + TensorFlow + scikit-learn.
Кожна лабораторна дає три результати: **код**, **навчальний конспект** (`GUIDE.md`) і **PDF-звіт**
за шаблоном кафедри.

## Швидкий старт

```bash
uv sync                  # один раз: створює .venv і ставить усі бібліотеки
uv run mllabs doctor     # перевірка: Python, TensorFlow, Chrome, титульна сторінка
```

Перед першим звітом заповніть `config/report.toml` (курс, група, ПІБ, викладач).

Виконати лабораторну — у Claude Code:

```
/lab 1
```

## Команди

| Команда | Що робить |
|---|---|
| `uv run mllabs list` | стан лабораторних: чи є код, конспект, PDF |
| `uv run mllabs new N` | створює каркас лабораторної №N із `labs/labNN/ASSIGNMENT.md` |
| `uv run mllabs run N` | виконує лабораторну; результати — у `labs/labNN/artifacts/` |
| `uv run mllabs report N` | збирає `labs/labNN/report/Звіт_ЛРN.pdf` із `report.md` |
| `uv run pytest` | тести |
| `uv run ruff format . && uv run ruff check .` | форматування й перевірка стилю |

`uv` — це водночас Maven і JDK-менеджер: `pyproject.toml` ≈ `pom.xml`, `uv.lock` — зафіксовані
версії, `.venv/` ≈ локальний репозиторій залежностей проєкту, `uv run X` — запуск у цьому оточенні.

## Що де лежить

```
ML_labs/
├── CLAUDE.md                  правила проєкту для агентів
├── config/report.toml         дані титульної сторінки звіту
├── labs/labNN/                МАТЕРІАЛИ лабораторної
│   ├── ASSIGNMENT.md            завдання (з методички)
│   ├── PLAN.md                  план і чекліст вимог
│   ├── GUIDE.md                 конспект: що, навіщо і як зроблено  ← читати першим
│   ├── artifacts/               графіки (PNG), таблиці (CSV), metrics.json
│   └── report/                  report.md → Звіт_ЛРN.pdf
├── src/mllabs/                КОД
│   ├── core/                    фреймворк: Lab, Step, LabContext, ArtifactStore
│   ├── reporting/               Markdown → HTML → PDF
│   ├── scaffold/                генератор каркаса лабораторної
│   └── labs/labNN/              код лабораторної №N
├── data/                      датасети (raw — сирі, processed — оброблені)
├── tests/                     тести (pytest)
├── docs/python-for-java.md    шпаргалка «Python для Java-розробника»
└── .claude/                   агенти та скіли
```

## Як влаштована лабораторна (коротко)

Кожна лабораторна — клас-нащадок `Lab` (патерн «Шаблонний метод»): він оголошує кроки, а базовий
клас виконує їх по черзі, логує час і фіксує випадковість.

```python
class Lab1(Lab):
    NUMBER = 1
    TITLE = "Дослідницький аналіз даних"

    def steps(self) -> Sequence[Step]:
        return [
            Step("1. Завантаження даних", self._load_data),
            Step("2. Аналіз кореляцій", self._analyze_correlations),
        ]
```

Усе, що має потрапити у звіт, код зберігає через `ArtifactStore` (`save_figure`, `save_table`,
`save_metrics`), а звіт лише посилається на ці файли. Тому цифри у звіті завжди збігаються з кодом.

## Агенти та скіли

| Агент | Роль | Результат |
|---|---|---|
| `lab-planner` | аналітик | `PLAN.md`: вимоги, вибір датасету й методів, дизайн |
| `ml-engineer` | розробник | код, тести, артефакти |
| `ml-reviewer` | рев'ювер (лише читає) | зауваження: витік даних, метрики, якість коду |
| `tutor` | наставник | `GUIDE.md` для Java-інженера |
| `report-writer` | автор звіту | `report.md` і PDF |

| Скіл | Призначення |
|---|---|
| `/lab N` | повний конвеєр лабораторної: план → код → рев'ю → конспект + звіт → приймання |
| `python-style` | стандарти коду |
| `lab-guide` | як писати конспект |
| `lab-report` | як писати звіт; синтаксис рисунків, таблиць, формул, метрик |

## Звіт

PDF друкує Google Chrome у headless-режимі (Word і LaTeX не потрібні). Оформлення відтворює
`Шаблон_звіту_ПР.docx` (титульна сторінка, структура розділів) і
`Шаблон_бакалаврської_роботи.dotm` (поля 2,5/1,5/1,5/1,5 см, Times New Roman 14, інтервал 1,5,
абзац 1,27 см, підписи рисунків і таблиць). Змінити оформлення —
`src/mllabs/reporting/templates/report.css`.
