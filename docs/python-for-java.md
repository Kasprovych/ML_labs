# Python для Java-розробника: шпаргалка проєкту

Лише те, що трапляється в цьому репозиторії. Конкретні приклади з кожної лабораторної — у розділі
«Python очима Java-розробника» відповідного `labs/labNN/GUIDE.md`.

## Інструменти

| Java | Python (тут) | Примітка |
|---|---|---|
| Maven / Gradle | `uv` | `uv sync` ≈ `mvn install`, `uv add x` ≈ додати `<dependency>` |
| `pom.xml` | `pyproject.toml` | залежності, налаштування лінтера й тестів |
| JDK-версія проєкту | `.python-version` | `uv` сам завантажить потрібний Python |
| `~/.m2` для проєкту | `.venv/` | ізольовані бібліотеки цього проєкту |
| Checkstyle + formatter | `ruff check` + `ruff format` | |
| JUnit | `pytest` | тест — звичайна функція `test_*`, перевірка — `assert` |
| `main(String[] args)` | `mllabs.cli:main` | оголошено в `[project.scripts]` |
| пакет = каталог | пакет = каталог з `__init__.py` | модуль = один `.py`-файл |

## Мова

| Java | Python |
|---|---|
| `public class A extends B implements C` | `class A(B):` — інтерфейси через `Protocol` (без `implements`) або `ABC` |
| `this.x` | `self.x` — `self` явно перший параметр кожного методу |
| конструктор `A(int x)` | `def __init__(self, x: int) -> None:` |
| `private` | домовленість: ім'я з `_` (`self._model`); компілятор не забороняє |
| `static final int N = 5` | `N = 5` на рівні модуля (`UPPER_CASE` = константа) |
| `record Point(int x, int y)` | `@dataclass(frozen=True) class Point: x: int; y: int` |
| `interface` | `class Renderer(Protocol): def render(...) -> None: ...` |
| `abstract` метод | `@abstractmethod` у класі-нащадку `ABC` |
| `@Override` | `@override` (необов'язково, але корисно) |
| `null` | `None`; перевірка `if x is None` |
| `Optional<T>` | `T | None` |
| `List<String>` / `Map<String,Integer>` | `list[str]` / `dict[str, int]` |
| `for (var x : items)` | `for x in items:` |
| `items.stream().map(f).filter(p).toList()` | `[f(x) for x in items if p(x)]` (list comprehension) |
| лямбда `x -> x + 1` | `lambda x: x + 1` |
| `try / catch / finally` | `try / except / finally` |
| try-with-resources | `with open(path) as stream:` |
| `String.format("%.2f", v)` | `f"{v:.2f}"` (f-рядок) |
| `throw new IllegalArgumentException()` | `raise ValueError(...)` |
| Javadoc `/** ... */` | docstring `"""..."""` першим рядком тіла |
| `{ }` | відступи: блок = 4 пробіли |

Типи в Python — **підказки**: їх перевіряє IDE та лінтер, а не інтерпретатор під час запуску.
Ми все одно пишемо їх скрізь — заради читабельності.

## Бібліотеки даних і ML

| Що | Чим є | Аналогія |
|---|---|---|
| NumPy `ndarray` | багатовимірний масив чисел із векторними операціями | `double[][]`, але `a + b` додає поелементно без циклів |
| pandas `DataFrame` | таблиця з іменованими колонками | результат SQL-запиту в пам'яті; `df[df.age > 30]` ≈ `WHERE` |
| pandas `Series` | одна колонка таблиці | `List<T>` з індексом |
| scikit-learn estimator | об'єкт із `fit(X, y)` → `predict(X)` | інтерфейс-стратегія: усі моделі взаємозамінні |
| transformer | об'єкт із `fit(X)` → `transform(X)` | двофазний: спершу «вивчає» параметри, потім застосовує |
| `Pipeline` | ланцюжок трансформерів + модель | `Function.andThen(...)`, що поводиться як одна модель |
| matplotlib / seaborn | побудова графіків | — |
| TensorFlow / Keras | нейронні мережі: шари → модель → `fit` | конструктор із блоків-шарів (Builder) |

Найважливіше правило, якого немає в Java-світі: **`fit` викликається тільки на навчальних даних**.
Якщо трансформер чи модель «побачить» тестові дані під час навчання, оцінка якості буде
завищеною — це називається витоком даних (data leakage), і йому присвячено ЛР №3.
