# План лабораторної роботи №3

Оптимізація гіперпараметрів (Grid / Random / Bayesian), дослідження витоку даних і вкладена
крос-валідація на датасеті Carvana `kick` (ЛР1) з моделлю з ЛР2.

Усі цифри в цьому плані отримано пробними запусками 2026-11-16 (scikit-learn 1.9.1, Optuna 5.0.0,
imbalanced-learn 0.14.2, Apple M1 Pro, 10 ядер = 8 P + 2 E) **на сильно завантаженій машині**
(load average від 47 до 440 через сторонні процеси). Це орієнтири для інженера, **не** цифри для
звіту: у звіт числа потрапляють лише з `artifacts/`. Одна проба (D7) порахувала PR-AUC на test;
жодне рішення плану за нею не налаштоване (див. розд. 5).

## 1. Вимоги (чекліст)

### 1.1. Вимоги із завдання

| ID | Вимога із завдання | Як буде виконано | Артефакт-доказ |
|---|---|---|---|
| R1 | Етап 1: підготовка даних | `ProcessedDatasetRepository.load(train_sample_size=20_000, random_state=seed)` — стратифікована підвибірка train ЛР1 як «робоча» вибірка (dev) для пошуку та вкладеної CV (D4); препроцесинг — сценарій Б (`PreprocessingScenario.FULL`) усередині пайплайна; `check_manifest` звіряє з ЛР1. Дослідження витоку — на повному train (D7) | `tables/data_state.csv`, `metrics.data.*` |
| R2 | Етап 1: створення baseline-моделі | HGB з параметрами ЛР2 (`ModelFactory().create(HIST_GRADIENT_BOOSTING)`) у пайплайні Б; оцінюється на тих самих 3 внутрішніх фолдах (порівнянно з Best Score) і на тих самих 5 зовнішніх фолдах (порівнянно з Nested); на test — як точка відліку | `tables/baseline.csv`, `metrics.baseline.*`, рядок «Baseline (ЛР2)» у `tables/optimization_methods.csv` |
| R3 | Обрати модель з ЛР2 з багатьма гіперпараметрами / схильну до перенавчання | `HistGradientBoostingClassifier` (D1): найкраща модель ЛР2, розрив train−CV ≈ 0,25, 6+ реальних гіперпараметрів, удвічі швидша за RF | `metrics.model.*`, `tables/baseline.csv` (train PR-AUC, розрив) |
| R4 | Зафіксувати метрику якості | PR-AUC (`average_precision`) — єдина метрика оптимізації в усіх методах (D2); ROC-AUC — допоміжна у зовнішній петлі, витоку і test | `metrics.metric.*` |
| R5 | Простір гіперпараметрів ≥ 3–4 параметрів | 4 параметри: `learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `max_features`; одні межі для всіх методів; сітка 3×3×3×3 = 81 (D3) | `tables/search_space.csv`, `metrics.search_space.*` |
| R6 | Full Grid Search (`GridSearchCV`) | `GridSearchCV`, усі 81 комбінації, 3 внутрішні фолди | рядок «Grid Search» у `tables/optimization_methods.csv`, `tables/search_history.csv` |
| R7 | Random Search, N = 20 | `RandomizedSearchCV(n_iter=20)` з неперервними / лог-рівномірними розподілами в тих самих межах | рядок «Random Search (N=20)» |
| R8 | Random Search, N = 50 | `RandomizedSearchCV(n_iter=50)`, той самий `random_state` (перші 20 кандидатів збігаються з N=20 — D5) | рядок «Random Search (N=50)» |
| R9 | Bayesian Optimization (Optuna), N = 20 | `optuna` + `TPESampler(seed=seed, multivariate=True, n_startup_trials=5)`, 20 спроб, ті самі межі й ті самі 3 фолди (D6) | рядок «Bayesian / Optuna TPE (N=20)» |
| R10 | Зафіксувати час виконання кожної стратегії | `time.perf_counter()` навколо кожного пошуку (wall-clock); додатково сумарний час навчань і час вкладеної CV | колонка «Час виконання (сек)», `metrics.methods.<m>.wall_seconds`, `...compute_seconds` |
| R11 | Зафіксувати найкраще значення валідаційної метрики | `best_score_` (середній PR-AUC на 3 внутрішніх фолдах) + його σ по фолдах; найкращі параметри | колонка «Best Score (Inner CV)», `tables/best_params.csv` |
| R12 | Некоректний пайплайн (з витоком) | Препроцесор Б навчається `fit(усі дані)` → `transform(усі дані)` **до** розбиття на фолди; модель — HGB ЛР2; витікає `TargetEncoder` (кожен рядок закодовано з його власною міткою) і всі статистики імп'ютерів/скейлерів (D7, варіант `leak_fit_all`) | рядок «Некоректний» у `tables/leakage_comparison.csv` |
| R13 | Коректний пайплайн (без витоку) | Той самий препроцесор і модель у `Pipeline`, `fit` лише на навчальній частині кожного фолда (`CrossValidator` з ЛР2) | рядок «Коректний» у `tables/leakage_comparison.csv` |
| R14 | Порівняти валідаційні метрики, оцінити величину завищення | Ті самі 5 фолдів → парна різниця Δ = некоректний − коректний з розкидом s (`paired_difference`); розклад за джерелами витоку (4 варіанти); залежність від обсягу даних (5 тис. / 20 тис. / 58 тис.); перевірка «реальністю» — обидва пайплайни на test з парним бутстреп-ДІ | `tables/leakage_comparison.csv`, `tables/leakage_folds.csv`, `tables/leakage_by_size.csv`, `figures/leakage_comparison.png`, `metrics.leakage.*` |
| R15 | Nested CV: зовнішня петля K = 5 | `StratifiedKFold(5, shuffle=True, random_state=seed)` на dev; ті самі зовнішні фолди для всіх методів і baseline | `tables/nested_folds.csv` |
| R16 | Nested CV: внутрішня петля K = 3 | У кожному зовнішньому фолді метод пошуку запускається на зовнішньому train з `StratifiedKFold(3, …)`; найкращі параметри → refit на зовнішньому train → оцінка на зовнішньому val | `tables/nested_folds.csv` (колонки `inner_best`, `outer_pr_auc`) |
| R17 | Порівняти оцінку внутрішньої CV і незміщену оцінку зовнішньої CV | Для кожного методу: оптимізм = inner best − outer по кожному зовнішньому фолду (парно), середнє ± σ; плюс порівняння Best Score самостійного пошуку з Nested-оцінкою | колонки таблиці методів, `figures/nested_inner_vs_outer.png`, `metrics.methods.<m>.optimism_*` |
| R18 | Звіт: таблиця «Метод оптимізації / Кількість ітерацій / оцінок / Час виконання (сек) / Best Score (Inner CV) / Unbiased Score (Nested CV)» | `tables/optimization_methods.csv` — рівно ці 5 колонок (заголовки дослівно); додаткові колонки — в `tables/optimization_cost.csv`, `tables/optimization_quality.csv`, `tables/nested_pairwise.csv` | `tables/optimization_methods.csv` |
| R19 | Звіт: таблиця/графік різниці метрик коректного й некоректного препроцесингу | І таблиця, і графік | `tables/leakage_comparison.csv`, `figures/leakage_comparison.png` |
| R20 | Звіт: графік «Якість моделі / Час обчислень» | Точки методів: x — wall-clock пошуку (лог-шкала), y — PR-AUC; порожній маркер — Best Score (Inner CV), заповнений — Nested ± σ; горизонтальна лінія baseline | `figures/quality_vs_time.png` |
| R21 | Звіт: висновки про доцільність методів за обмежених ресурсів | Звіт спирається на `optimization_cost.csv` (навчань/с, процесів × потоків, wall-clock), `optimization_quality.csv` (приріст над baseline з s, оптимізм сирий і відбору), `nested_pairwise.csv` (Δ > s?), криві збіжності і test; правило висновку — D9 | `figures/search_convergence.png`, `metrics.methods.*`, `tables/test_results.csv` |

### 1.2. Проєктні вимоги (з `CLAUDE.md`, мети роботи і залежностей)

| ID | Вимога | Як буде виконано | Артефакт-доказ |
|---|---|---|---|
| P1 | Відтворюваність | Усі зерна з `LabContext.seed`: підвибірка, фолди, HGB `random_state`, `RandomizedSearchCV.random_state`, `TPESampler(seed=…)`, SMOTE, бутстреп. Optuna — спроби строго послідовно (з `n_jobs > 1` TPE невідтворюваний). Тест: два запуски Optuna на мініданих дають однакову історію | `metrics.config.*`, тести |
| P2 | `fit` лише на навчальних даних (коректна частина) | Пошук і вкладена CV — лише через `Pipeline` (препроцесор усередині). Некоректні варіанти живуть в одному модулі `leakage.py`, мають «гучні» назви й docstring і ніде більше не використовуються | тест «зовнішній val не потрапляє у внутрішній train», тест «leak-варіант справді протікає» |
| P3 | Перевикористання спільного коду | `PreprocessorFactory`, `build_model_pipeline`, `SamplerFactory` (ЛР1), `ModelFactory`, `CrossValidator`, `MetricsCalculator`, `PairedBootstrap` (ЛР2), `paired_difference` (ЛР1). Нічого не імпортується з `mllabs.labs.lab01/lab02` | огляд коду |
| P4 | Тривалість `mllabs run 3` — хвилини | Бюджет ≈ 20 хв на незавантаженій машині (розд. 5.1); підвибірка і паралелізм — у `config.py` | `metrics.timing.*` |
| P5 | Мета: «вплив алгоритму пошуку на обчислювальну складність і якість» | Крім таблиці — криві збіжності (best-so-far за номером оцінки) і зведення «оцінок / с», «секунд на 0,001 PR-AUC приросту» | `figures/search_convergence.png`, `tables/optimization_cost.csv` |
| P6 | Чесність фінальної оцінки | Test використовується один раз для заздалегідь визначеного переліку (baseline + 4 налаштовані моделі + 2 пайплайни витоку); нічого за ним не обирається (D8) | `tables/test_results.csv` |

## 2. Ключові рішення

### D1. Модель

Дані ЛР2 (`metrics.lab3_handoff`, сценарій Б, 5-кратна CV на повному train): HGB — PR-AUC 0,485,
train 0,807 (розрив 0,323; до виправлення помилки ЛР2 — 0,730 / 0,245), ~5,6 с на фолд; RF — 0,469, train 0,999 (розрив 0,530), ~11,6 с на фолд.

| Варіант | Плюси | Мінуси |
|---|---|---|
| 1. **HistGradientBoosting** | найкраща модель ЛР2; багато реальних важелів (`learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `max_features`, `l2_regularization`, `max_iter`); помітний розрив train/CV; удвічі швидша — без цього Grid 81 у вкладеній CV не вкладається в бюджет | розрив менш «драматичний», ніж у RF |
| 2. RandomForest | найнаочніше перенавчання (train ≈ 1,0) | ~2× дорожчий на фолд; ліс зазвичай мало виграє від тюнінгу (`max_features`, `min_samples_leaf`), а час росте з `n_estimators`; Grid 81 × вкладена CV ≈ 40+ хв |
| 3. Обидві | повніше | подвоює найдорожчу частину; таблиця завдання — для однієї моделі |

**Рекомендація: варіант 1** (ПОТРЕБУЄ ПОГОДЖЕННЯ — рекомендацію прийнято за замовчуванням).
Baseline — HGB з точними параметрами ЛР2 (`max_iter=200`, `learning_rate=0.1`, `max_leaf_nodes=31`,
`min_samples_leaf=20`, `l2_regularization=0`, `early_stopping=False`). `max_iter=200` і
`early_stopping=False` фіксуються і в пошуку: (1) кількість дерев явна і однакова, (2) раннє
зупинення само відрізало б 10 % даних і змішувало б ефект параметрів, (3) час однієї оцінки
передбачуваний.

### D2. Метрика

| Варіант | Плюси | Мінуси |
|---|---|---|
| 1. **PR-AUC (`average_precision`)** | головна метрика ЛР1–2; чутлива до якості саме на рідкісному класі (12 %); не залежить від порога | менш відома, ніж ROC-AUC |
| 2. ROC-AUC | відома; не залежить від частки класів | при 12 % позитивних «оптимістична» — слабко розрізняє моделі (у ЛР2 HGB Б 0,772 проти LR Б 0,759) |
| 3. F1 | є в завданні як приклад | залежить від порога; оптимізація F1 = оптимізація порога + параметрів одночасно |

**Рекомендація: варіант 1.** Усі чотири методи оптимізують `scoring="average_precision"`
(Optuna — `sklearn.metrics.average_precision_score` на тих самих фолдах). ROC-AUC рахується
додатково лише там, де вона потрібна для інтерпретації (зовнішня петля, витік — особливо для
варіанта зі SMOTE, де змінюється частка класів, test). Поріг/F1 у ЛР3 не підбираються — це ЛР2.

### D3. Простір гіперпараметрів

Проба (Grid 2⁴ = 16 комбінацій × 3 фолди на 20 тис.): найкращі комбінації — з `learning_rate=0.03`
(нижня межа проби), `0.3` стабільно гірше; 63 листки на ~20 % повільніші за 15; `max_features`
на час майже не впливає. Тобто оптимум за `learning_rate` лежить нижче 0,1, і межі мають
його охоплювати.

| Параметр | Що робить (для звіту) | Сітка (Grid) | Межі для Random / Optuna | Шкала |
|---|---|---|---|---|
| `learning_rate` | внесок кожного дерева; менший — повільніше вчиться, менше перенавчається | 0,01; 0,03; 0,1 | [0,01; 0,1] | лог, float |
| `max_leaf_nodes` | складність одного дерева | 15; 31; 63 | [15; 63] | лог, int |
| `min_samples_leaf` | мінімум рядків у листку — регуляризація | 20; 100; 500 | [20; 500] | лог, int |
| `max_features` | частка ознак, яку бачить кожне розбиття (як у випадковому лісі) | 0,3; 0,6; 1,0 | [0,3; 1,0] | лінійна, float |

| Варіант «однаковості простору» | Плюси | Мінуси |
|---|---|---|
| 1. **Однакові межі: Grid — 3 точки на вісь, Random/Optuna — неперервні розподіли в тих самих межах** | класичне порівняння (Bergstra & Bengio): Random бачить 20–50 різних значень кожного параметра, Grid — лише 3; це і є суть відмінності методів | простори формально не тотожні (дискретний і неперервний) — треба пояснити у звіті |
| 2. Усі три методи на тій самій дискретній сітці 81 точок | ідеально однаковий простір | Random = Grid без частини точок; TPE на категоріальних параметрах слабкий; втрачається головна перевага Random/Bayesian |

**Рекомендація: варіант 1.** Один об'єкт `SearchSpace` — єдине джерело меж; з нього генеруються
і `param_grid`, і розподіли `RandomizedSearchCV`, і виклики `trial.suggest_*`, тож розбіжність меж
неможлива (тест). Сітка завжди містить обидві межі. Відкинуті: `l2_regularization` (у HGB слабкий
ефект; нуль не вкладається в лог-шкалу); `max_iter` (лінійно множить час і дублює `learning_rate`);
п'ятий параметр (сітка 243 не вкладається в бюджет). Розміри: 81 (Grid) > 50 > 20 = 20.

### D4. Обсяг даних для пошуку та вкладеної CV

Кількість навчань моделі: (81 + 20 + 50 + 20) кандидатів × 3 фолди × (1 самостійний пошук +
5 зовнішніх фолдів) = **3 078** навчань пайплайна + 25 refit. Проба на 20 тис.: одне навчання
(13,3 тис. рядків, 1 потік) ≈ 2,9 с; `GridSearchCV(n_jobs=10)` — ≈ 0,37 с wall на одне навчання
під залишковим навантаженням (load ≈ 47).

| Варіант | Позитивних у зовнішньому val | Оцінка часу всього запуску | Плюси | Мінуси |
|---|---|---|---|---|
| 1. Повний train (58 386) | ~1 440 | ≈ 60–70 хв | найточніше | у 3× за бюджетом |
| 2. 30 000 | ~740 | ≈ 28–32 хв | точніше | понад ціль |
| 3. **20 000** | ~490 | **≈ 18–22 хв** | вкладається в ціль; ~2 460 позитивних у dev | σ PR-AUC зовнішнього фолда ≈ 0,02 — різниця між методами, найпевніше, у межах шуму (так і писати) |
| 4. 15 000 | ~370 | ≈ 14–16 хв | запас часу | ще шумніше |

**Рекомендація: варіант 3, 20 000 рядків** (ПОТРЕБУЄ ПОГОДЖЕННЯ — тривалі обчислення).
Пошук (етап 2), вкладена CV (етап 4), baseline і фінальні моделі для test — на **тій самій**
dev-вибірці: інакше Best Score і Nested-оцінка відносились би до різних обсягів даних і їх не
можна було б порівнювати. Правило калібрування для інженера: після першого запуску
`загальний час ≈ 14 × час самостійного Grid Search + ~3 хв`; якщо прогноз > 25 хв — 15 000; якщо
< 12 хв — 30 000. Обраний розмір і заміряний час записати в docstring `config.py`. Дослідження
витоку цієї підвибірки не потребує (D7).

Чесне обмеження для звіту: HGB, налаштований на 20 тис., не порівнюється напряму з HGB ЛР2 на
58 тис. — тому baseline у ЛР3 теж рахується на тих самих 20 тис.

### D5. Протокол пошуку, вкладеної CV і вимірювання часу

1. **Самостійний пошук (етап 2)**: кожен метод на всій dev-вибірці, внутрішні фолди
   `StratifiedKFold(3, shuffle=True, random_state=seed)` — **один і той самий список розбиттів** для
   всіх методів (передається явно як `cv=list(splits)`). `refit=False`: час — лише оцінювання
   кандидатів; фінальний refit робиться окремо (D8). `error_score="raise"`.
   - «Кількість ітерацій / оцінок»: `81 / 243`, `20 / 60`, `50 / 150`, `20 / 60` (кандидатів /
     навчань моделі).
   - «Best Score (Inner CV)» = `best_score_` (середнє по 3 фолдах) самостійного пошуку; σ по фолдах
     — у деталях.
2. **Вкладена CV (етап 4)**: зовнішні `StratifiedKFold(5, shuffle=True, random_state=seed)` на dev
   (ті самі для всіх методів і baseline). У кожному зовнішньому фолді: той самий метод пошуку на
   зовнішньому train з `StratifiedKFold(3, shuffle=True, random_state=seed)` → найкращі параметри →
   `clone(pipeline).set_params(**best)` → `fit` на зовнішньому train → PR-AUC і ROC-AUC на зовнішньому
   val. Зовнішні фолди виконуються послідовно, паралелізм — усередині пошуку (як у п. 1), тому
   режим виконання однаковий з етапом 2.
   - «Unbiased Score (Nested CV)» = середній зовнішній PR-AUC ± σ (5 фолдів).
   - Оптимізм = `inner_best_k − outer_k` по фолдах (парно), середнє ± σ; додатково
     «Best Score самостійного пошуку − Nested».
   - Приріст від тюнінгу = `outer_k(метод) − outer_k(baseline)` на тих самих зовнішніх фолдах,
     `paired_difference` (Δ, s).
3. **Паралелізм і час.** Grid/Random: `n_jobs=-1` (паралельно по кандидатах×фолдах, 10 процесів,
   joblib сам обмежує потоки OpenMP у кожному до `10 // n_jobs`). Optuna: спроби послідовно
   (суть TPE і відтворюваність), а 3 фолди спроби — паралельно (`joblib.Parallel(n_jobs=3)`,
   ~3 потоки OpenMP на процес). Обидва режими використовують ≈ всі ядра, тому wall-clock чесно
   порівнюваний; різниця «паралелиться легко / послідовний за природою» — сама по собі аргумент
   для висновків. Записується: `wall_seconds` (головне), `compute_seconds` (сума часу навчання +
   оцінювання всіх фолдів; з `cv_results_` або замірів Optuna), `nested_wall_seconds`.
   Попередження: на завантаженій машині HGB з усіма потоками OpenMP сповільнюється в рази (у ЛР2:
   271 с проти 8 с; у пробі ЛР3: 54 с проти 2,5 с на одне навчання) — головні запуски робити на
   вільній машині; для фінальних refit у головному процесі обмежити потоки
   `threadpoolctl.threadpool_limits(config.main_process_threads)` (пакет уже встановлений як
   залежність sklearn).

| Варіант паралелізму Optuna | Плюси | Мінуси |
|---|---|---|
| 1. **Послідовні спроби, 3 фолди паралельно** | відтворювано з `seed`; чистий TPE | лише 3 процеси (але по ~3 потоки) |
| 2. `study.optimize(n_jobs=10)` | швидше | невідтворювано; 10 паралельних спроб із 20 — фактично випадковий пошук |
| 3. Обмежити Grid/Random теж до 3 процесів | ідентичний режим | загальний час ≈ 30 хв |

**Рекомендація: варіант 1.**

*Чи доречно RS20 як префікс RS50.* `RandomizedSearchCV` з тим самим цілим `random_state` і
неперервними розподілами генерує ту саму послідовність, тож перші 20 кандидатів RS50 = RS20
(перевірити тестом). Це **бажано**: різниця між рядками таблиці — рівно «ще 30 ітерацій», а не
везіння іншого зерна. Обидва пошуки все одно виконуються окремо (окремий заміряний час).

### D6. Bayesian Optimization (Optuna)

`TPESampler(seed=seed, multivariate=True, n_startup_trials=5)`, `direction="maximize"`, без прунера,
20 спроб; `optuna.logging.set_verbosity(WARNING)`; `ExperimentalWarning` про `multivariate`
перехоплюється і ігнорується явно (з коментарем).

| Варіант `n_startup_trials` | Плюси | Мінуси |
|---|---|---|
| 1. Типове 10 | стандарт Optuna | половина бюджету N = 20 — чисто випадкові точки; «Bayesian» ≈ RS10 + 10 кроків TPE |
| 2. **5** | 15 із 20 спроб керовані моделлю; 5 точок на 4 виміри — мінімально розумний старт | менше «розвідки» на початку |

**Рекомендація: варіант 2.** `multivariate=True` — TPE моделює параметри спільно (важливо для пари
`learning_rate` × `max_leaf_nodes`). Прунер (`MedianPruner`) не використовується: кожна спроба має
проходити ті самі 3 фолди, інакше «кількість оцінок» не порівнянна з Grid/Random; згадати у
висновках як спосіб подальшої економії. `BayesSearchCV` (scikit-optimize) не встановлено і не
додається. `optuna-integration` (`OptunaSearchCV`) не встановлено — власний тонкий цикл (розд. 3)
простіший і дає ту саму схему, що й sklearn-пошуки.

### D7. Дослідження витоку даних

Проба (повний train, 5 фолдів, HGB ЛР2, ті самі фолди):

| Варіант | PR-AUC CV | σ | PR-AUC на train-фолдах |
|---|---|---|---|
| коректний (`Pipeline`) | 0,485 | 0,008 | 0,730 |
| препроцесор `fit_transform(усі)` (TE з крос-фітингом, решта — на всіх даних) | 0,487 | 0,006 | 0,804 |
| препроцесор `fit(усі).transform(усі)` | **0,521** | 0,009 | 0,796 |
| те саме на 20 тис. рядків: коректний → некоректний | 0,464 → 0,545 | Δ = 0,081 ± 0,020 | — |
| SMOTE до розбиття, val лише реальні рядки | 0,478 | 0,014 | — |
| SMOTE до розбиття, val з синтетичними рядками (як зробив би новачок) | 0,925 | — | — |

Висновки проби: (1) `TargetEncoder`, навчений на всіх даних, дає стабільне завищення в кожному
фолді (+0,025…+0,048 на повному train) і тим більше, чим менше даних (рідкісні `Model`/`SubModel`:
~450 категорій з ≤ 5 рядків); (2) «безучительські» кроки (медіана, Yeo-Johnson, скейлер,
агрегати, One-Hot) на всіх даних практично не завищують — це теж результат для звіту; (3) SMOTE
до розбиття для HGB не завищує оцінку на **реальних** рядках, але дає фантастичні 0,93, якщо
валідувати і на синтетичних (зміщена частка класів + синтетичні точки — інтерполяції сусідів з
train).

**Варіанти (усі — HGB ЛР2, ті самі 5 фолдів `CrossValidator.splits`, повний train):**

| Ключ | Назва в звіті | Що витікає | Як реалізується | Порівнюється з |
|---|---|---|---|---|
| `correct` | **Коректний** | нічого | `build_model_pipeline(FULL, HGB)` у `CrossValidator` | — |
| `leak_unsupervised` | Препроцесинг на всіх даних (`fit_transform`) | медіани, параметри Yeo-Johnson/скейлера, агрегати, словники One-Hot; TE — крос-фітинг по всіх даних | `Z = pre.fit_transform(X, y)` один раз → CV лише моделі на `Z` | `correct` |
| `leak_fit_all` | **Некоректний** (головний) | усе з попереднього + TE кодує кожен рядок статистикою з його власною міткою | `pre.fit(X, y)`; `Z = pre.transform(X)` (саме так, не `fit_transform` — див. docstring `_high_cardinality_pipeline`) → CV моделі | `correct` |
| `correct_smote` | Коректний + SMOTE у пайплайні | нічого | `build_model_pipeline(FULL, HGB, SMOTE(0.5))` (як сценарій В ЛР2) | — |
| `leak_smote` | SMOTE до розбиття | синтетичні точки, побудовані з val-позитивів, потрапляють у train | `Z` з `leak_unsupervised` → `SMOTE(0.5, k=5, seed).fit_resample` → реальні рядки зберігають свої фолди, синтетичні розподіляються по фолдах випадково (seed) → навчання на train-частині (реальні + синтетичні) → оцінка: (а) лише реальні val-рядки (головна), (б) val із синтетичними («як новачок») | `correct_smote` |

Для кожного варіанта: PR-AUC і ROC-AUC по фолдах, PR-AUC на train-фолдах (розрив), Δ проти
пари і s (`paired_difference`). Для `leak_smote` (б) порівнювати лише ROC-AUC і прямо написати,
що PR-AUC при частці класу 33 % проти 12 % непорівнянна.

**Перевірка «реальністю» (test):** `correct` і `leak_fit_all` навчаються на всьому train (для
некоректного: `pre.fit(train)` → `transform(train)` → модель; test кодується статистикою train —
як буде «в продакшні») → PR-AUC на test + парний бутстреп Δ (1000 повторів). Величина штучного
завищення = `CV(leak_fit_all) − test(leak_fit_all)`; додатково видно, що модель з витоком на нових
даних **гірша** за коректну (вона навчилась довіряти «підглянутому» кодуванню).

**Залежність від обсягу:** `correct` проти `leak_fit_all` на стратифікованих підвибірках
5 000 і 20 000 (20 000 — та сама dev-вибірка, що в пошуку) + повний train → `leakage_by_size.csv`.

| Варіант обсягу дослідження | Плюси | Мінуси |
|---|---|---|
| 1. Лише «коректний vs некоректний» | буквально завдання | не пояснює, ЩО саме протікає і чому завищення саме таке |
| 2. **Розклад на 4 варіанти + обсяг даних + test** (~2 хв) | кожне твердження звіту має число з розкидом; видно, що не кожен «витік» шкідливий однаково | більше таблиць |
| 3. + відбір ознак за ціллю на всіх даних (`SelectKBest`) | ще одне класичне джерело | у рецепті Б немає відбору за ціллю — довелося б штучно додати крок; при 92 інформативних ознаках ефект слабкий і нічого не пояснює про НАШ пайплайн |

**Рекомендація: варіант 2** (ПОТРЕБУЄ ПОГОДЖЕННЯ: що вважати «некоректним пайплайном» у таблиці
звіту — рекомендовано `leak_fit_all` без SMOTE, бо він порівнюваний за PR-AUC; SMOTE — окремим
рядком розкладу). Дослідження витоку — на **повному** train: воно дешеве (≈ 30 навчань) і
реалістичніше; 20 тис. — у розрізі обсягу.

### D8. Фінальна оцінка на test

Test уже «спалений» у ЛР2 для 12 фіксованих конфігурацій. Повторне використання доречне за
умови, що за ним **нічого не обирається**: перелік моделей визначено заздалегідь, кожна
оцінюється один раз.

| Варіант | Плюси | Мінуси |
|---|---|---|
| 1. Не використовувати test | максимально «чисто» | нема підтвердження, що Nested-оцінка реалістична; завдання цього не вимагає, але висновки слабші |
| 2. **Кожен із 4 методів: найкращі параметри самостійного пошуку → refit на dev (20 тис.) → test; baseline так само; парний бутстреп «метод − baseline»** | Nested-оцінка порівнюється з незалежною; без вибору «найкращого за test» | test той самий, що в ЛР2 — у звіті назвати; 4 налаштовані моделі майже однакові — різниці в межах ДІ |
| 3. Refit на повному train (58 тис.) | сильніша модель | гіперпараметри (напр. `min_samples_leaf`) підібрано під 20 тис.; Nested-оцінка відноситься до 20 тис. — порівняння зміщене |

**Рекомендація: варіант 2.** У звіті явно: «test не впливав на жоден вибір; ЛР2 використала його
для інших конфігурацій; різниці між методами інтерпретуємо лише разом із бутстреп-ДІ».
Опціонально для довідки (без обчислень) — PR-AUC test HGB ЛР2 (58 тис.) з
`labs/lab02/artifacts/metrics.json` записується в `metrics.lab2_reference` (якщо файл є; інакше
WARNING і пропуск).

### D9. Графік «Якість / Час» і правило висновків

- `quality_vs_time.png` (R20): x — wall-clock самостійного пошуку, с (лог-шкала); y — PR-AUC.
  Для кожного методу: порожній маркер — Best Score (Inner CV), заповнений — Nested ± σ (вертикальна
  риска); підпис «N кандидатів / навчань»; горизонтальна пунктирна лінія — baseline на тих самих
  зовнішніх фолдах (± σ смугою). Одним поглядом видно і вартість, і оптимізм внутрішньої оцінки.
- `search_convergence.png` (P5): best-so-far внутрішнього PR-AUC від номера оцінки (кандидата)
  для 4 методів (Grid — у порядку `ParameterGrid`, тому крива сходинкова — пояснити); для Optuna
  вертикальна риска на межі 5 стартових випадкових спроб.
- **Правило висновку** (без «гарних» тверджень): метод вважається кращим за інший, лише якщо парна
  різниця зовнішніх PR-AUC Δ > s (те саме правило, що в ЛР1–2). Якщо різниці в межах шуму (очікувано)
  — висновок про доцільність робиться за часом: «та сама якість за X с замість Y с».

## 3. Дизайн коду

### 3.1. Спільні пакети

Нових спільних модулів не потрібно. Використовуються як є:
`mllabs.data.ProcessedDatasetRepository` (підвибірка), `mllabs.preprocessing.{PreprocessorFactory,
PreprocessingScenario, build_model_pipeline, MODEL_STEP, SamplerFactory, SamplerKind,
paired_difference}`, `mllabs.modeling.{ModelFactory, ModelKind, CrossValidator, MetricsCalculator,
PairedBootstrap, positive_proba, mean_and_std}`.

Якщо на рев'ю ЛР2 зміниться API `CrossValidator`/`PairedBootstrap` — адаптувати виклики, логіку
ЛР3 не змінювати. Параметри пошуку адресуються як `f"{MODEL_STEP}__{name}"` (константа з ЛР1),
а не рядком `"model__..."`.

Код пошуку (Grid/Random/Optuna) лишається в `labs/lab03`: ЛР4–5 — нейромережі на TensorFlow,
перевикористання не передбачається.

### 3.2. Пакет `src/mllabs/labs/lab03/`

| Файл | Клас / функція | Відповідальність |
|---|---|---|
| `config.py` | `Lab3Config` (frozen) | Усі числа: див. 3.4 |
| `search_space.py` | `ParameterRange` (frozen: `name`, `low`, `high`, `grid`, `log: bool`, `integer: bool`; перевірка в `__post_init__`: `grid` ⊂ [`low`, `high`], обидві межі в сітці), `SearchSpace` (frozen, кортеж діапазонів), `IntLogUniform` | `param_grid(prefix)` для `GridSearchCV`; `distributions(prefix)` для `RandomizedSearchCV` (`scipy.stats.loguniform`, `uniform`, власний `IntLogUniform` з методом `rvs(size=None, random_state=None)` → `int`); `suggest(trial)` для Optuna (`suggest_float/int(..., log=...)`); `grid_size`; `describe()` → рядки для `search_space.csv` |
| `search.py` | `SearchMethod` (`StrEnum`: `GRID`, `RANDOM_20`, `RANDOM_50`, `OPTUNA_20`), `EvaluationRecord` (frozen: `index`, `params`, `mean_score`, `std_score`, `fold_scores`, `compute_seconds`), `SearchOutcome` (frozen: `method`, `n_candidates`, `n_fits`, `wall_seconds`, `compute_seconds`, `best_params`, `best_score`, `best_score_std`, `history: tuple[EvaluationRecord, ...]`; `best_so_far()`), `HyperparameterSearch` (`Protocol`: `method`, `run(pipeline, X, y, splits) -> SearchOutcome`), `GridSearchStrategy`, `RandomSearchStrategy(n_iter)`, `OptunaSearchStrategy(n_trials)`, `SearchStrategyFactory` | Одна «стратегія» = один метод пошуку з однаковим інтерфейсом (патерн Strategy, як інтерфейс у Java). sklearn-стратегії: `refit=False`, `cv=splits`, `scoring` з конфігу, історія з `cv_results_` (`params`, `mean_test_score`, `split{k}_test_score`, `mean_fit_time+mean_score_time` × k). Optuna: `create_study(direction="maximize", sampler=TPESampler(...))`, objective = `suggest` → `clone(pipeline).set_params(...)` → паралельно 3 фолди → середній PR-AUC; fold-скори і час — у `trial.set_user_attr`. Час — `perf_counter` навколо всього пошуку |
| `nested.py` | `NestedFoldResult` (frozen: `fold`, `inner_best`, `outer_pr_auc`, `outer_roc_auc`, `best_params`, `search_wall_seconds`, `refit_seconds`), `NestedResult` (frozen; `mean`, `std`, `optimism()` → (`mean`, `std`), `fold_scores`), `NestedCrossValidation` | Зовнішній цикл: для кожного зовнішнього фолда — внутрішні розбиття на зовнішньому train, `strategy.run(...)`, refit, оцінка на зовнішньому val. Метод `outer_splits(X, y)` (однакові для всіх методів і baseline) |
| `baseline.py` | `BaselineEvaluator`, `BaselineResult` (frozen) | HGB ЛР2 на 3 внутрішніх розбиттях dev (порівнянно з Best Score) і на 5 зовнішніх (порівнянно з Nested, з train PR-AUC для розриву) — через `CrossValidator` або той самий код оцінки фолда, що в `nested.py` |
| `leakage.py` | `LeakageVariant` (`StrEnum`, D7), `LeakageFoldScores` (frozen), `LeakageResult` (frozen: варіант, fold PR-AUC/ROC-AUC, train PR-AUC, для SMOTE — ще naive-оцінки), `LeakyMatrixBuilder`, `SmoteBeforeSplitEvaluator`, `LeakageExperiment`, `LeakageHoldoutCheck` | **Єдине місце з навмисно некоректним кодом.** `LeakyMatrixBuilder.fit_transform_all(...)` / `.fit_all_then_transform(...)` — docstring з поясненням, чому це витік. Коректні варіанти — через `CrossValidator.evaluate`; матричні — `CrossValidator.evaluate(clone(model), pd.DataFrame(Z), y)` (ті самі фолди, бо `splits` залежить лише від `y` і seed); SMOTE до розбиття — власний цикл по `CrossValidator.splits`. `LeakageHoldoutCheck`: обидва пайплайни на повному train → test, `PairedBootstrap`. Розріз за обсягом — той самий `LeakageExperiment` на підвибірках |
| `holdout.py` | `FinalModelEvaluator`, `HoldoutRow` (frozen) | Для baseline і 4 методів: `set_params(best)` → `fit(dev)` → `predict_proba(test)` → PR-AUC, ROC-AUC; бутстреп «метод − baseline» |
| `summary.py` | `MethodsTableBuilder`, `DetailsTableBuilder`, `LeakageTableBuilder`, функції `search_history_frame`, `nested_folds_frame`, `best_params_frame`, `metrics_payload` | Таблиці з **дослівними** заголовками завдання; форматування «81 / 243»; формування словника для `metrics.json` |
| `plots.py` | `plot_quality_vs_time`, `plot_search_convergence`, `plot_nested_inner_vs_outer`, `plot_leakage_comparison`, `plot_leakage_by_size`, `plot_search_space_samples` | Графіки зі стилем `mllabs.plotting.style`; підписи українською |
| `lab.py` | `Lab3(Lab)` | Лише оркестрація кроків |

### 3.3. Кроки `Step` у `lab.py`

1. **«1. Дані, метрика та baseline»** — `repo.load(train_sample_size=dev_size, random_state=seed)`
   (dev + повний test), окремо `repo.load()` (повний train для витоку); `check_manifest`;
   `data_state.csv`; `search_space.csv`; baseline на 3 внутрішніх і 5 зовнішніх розбиттях →
   `baseline.csv`; `metrics.data/metric/model/baseline/search_space`.
2. **«2. Дослідження 1: Grid, Random, Bayesian»** — 4 самостійні пошуки на dev у порядку
   Grid → RS20 → RS50 → Optuna; `search_history.csv`, `best_params.csv`, проміжний
   `metrics.methods.<m>.{n_candidates, n_fits, wall_seconds, compute_seconds, best_score_inner, ...}`.
3. **«3. Дослідження 2: витік даних»** — 5 варіантів на повному train; розріз за обсягом
   (5 тис., 20 тис.); перевірка на test; `leakage_comparison.csv`, `leakage_folds.csv`,
   `leakage_by_size.csv`, `metrics.leakage.*`.
4. **«4. Дослідження 3: вкладена крос-валідація 5×3»** — для кожного методу (той самий порядок);
   `nested_folds.csv`, `metrics.methods.<m>.{nested_*, optimism_*, gain_vs_baseline_*}`.
5. **«5. Фінальна оцінка на test»** — baseline + 4 моделі; `test_results.csv`,
   `metrics.methods.<m>.test_*`, `metrics.lab2_reference` (якщо є).
6. **«6. Зведені таблиці та графіки»** — `optimization_methods.csv`, `optimization_cost.csv`, `optimization_quality.csv`, `nested_pairwise.csv`,
   усі графіки, `metrics.timing.*`, `metrics.config`.

Кожен крок одразу пише свої артефакти (якщо запуск обірветься на кроці 4, результати 1–3 вже на
диску). Етап витоку стоїть третім (порядок завдання), хоча від пошуку не залежить.

### 3.4. Параметри `config.py` (`Lab3Config`)

| Поле | Значення | Примітка |
|---|---|---|
| `dataset_name` | `"kick"` | |
| `model` | `ModelKind.HIST_GRADIENT_BOOSTING` | D1 |
| `model_settings` | `ModelSettings()` | параметри ЛР2 = baseline |
| `preprocessing` | `PreprocessingSettings()` | звіряється з ЛР1 |
| `scoring` | `"average_precision"` | D2 |
| `dev_sample_size` | `20_000` | D4; калібрується за правилом D4 |
| `inner_folds` | `3` | завдання |
| `outer_folds` | `5` | завдання |
| `search_space` | `SearchSpace` з 4 `ParameterRange` (D3) | `default_factory` |
| `random_iterations` | `(20, 50)` | завдання |
| `optuna_trials` | `20` | завдання |
| `optuna_startup_trials` | `5` | D6 |
| `optuna_multivariate` | `True` | D6 |
| `search_n_jobs` | `-1` | Grid/Random |
| `optuna_fold_n_jobs` | `3` | = `inner_folds` |
| `main_process_threads` | заміряти (орієнтир 8) | `threadpool_limits` для refit у головному процесі |
| `leakage_folds` | `5` | як у ЛР1–2 |
| `leakage_cv_n_jobs` | `5` | як `Lab2Config.cv_n_jobs` |
| `leakage_sample_sizes` | `(5_000, 20_000)` | + повний train |
| `sampling_strategy`, `smote_neighbors`, `noise_scale` | `0.5`, `5`, `0.05` | як ЛР1–2 (`noise_scale` лише для конструктора `SamplerFactory`) |
| `bootstrap_repeats`, `confidence_level` | `1000`, `0.95` | як ЛР2 |

### 3.5. Тести (`tests/labs/test_lab03_*.py`)

- `SearchSpace`: сітка = 81 точка; кожне значення сітки в межах; розподіли Random і `suggest`
  Optuna (на `optuna.trial.FixedTrial` / мінідослідженні) не виходять за межі; цілі параметри — `int`.
- `RandomSearchStrategy`: перші 20 кандидатів N=50 збігаються з N=20 (на мінізадачі).
- `OptunaSearchStrategy`: два запуски з тим самим seed → однакова історія параметрів.
- `NestedCrossValidation`: шпигун-стратегія фіксує індекси, на яких її запущено, — жоден рядок
  зовнішнього val не потрапляє у внутрішній пошук; `inner_best`/`outer` записані для кожного фолда.
- `leakage`: на синтетичних даних, де категорія-ідентифікатор унікальна для кожного рядка,
  `leak_fit_all` дає PR-AUC ≈ 1, а `correct` ≈ частці класу; `leak_unsupervised` не підглядає власну
  мітку. SMOTE до розбиття: реальні рядки зберігають фолди, кількість синтетичних = очікуваній.
- `summary`: `optimization_methods.csv` має рівно 5 колонок із дослівними заголовками завдання.
- `Lab3` end-to-end на крихітних фікстурах (як `test_lab02_lab.py`) з мінімальним конфігом
  (сітка 2×2, N=3, 2×2 фолди) — перевірка, що всі артефакти з розд. 4 створюються.

## 4. Артефакти (`labs/lab03/artifacts/`)

### 4.1. Графіки (`figures/`)

| Файл | Зміст | Вимога |
|---|---|---|
| `quality_vs_time.png` | Якість (PR-AUC: Inner — порожні, Nested ± σ — заповнені) проти wall-clock пошуку (лог), baseline-лінія | R20 |
| `search_convergence.png` | Best-so-far внутрішнього PR-AUC від номера оцінки, 4 методи | P5, R21 |
| `nested_inner_vs_outer.png` | По кожному методу 5 пар «inner best → outer» (гантелі) — видно оптимізм | R17 |
| `leakage_comparison.png` | Стовпчики PR-AUC CV ± σ для 5 варіантів витоку + точки test для `correct`/`leak_fit_all`; підписи Δ ± s | R19, R14 |
| `leakage_by_size.png` | Δ (некоректний − коректний) ± s від обсягу даних (5 тис., 20 тис., 58 тис.) | R14 |
| `search_space_samples.png` | Значення кожного параметра, які перебрали Grid / RS50 / Optuna, колір — внутрішній PR-AUC (видно: Grid — 3 значення, Random — розкид, TPE — згущення) | R5, R21 |

### 4.2. Таблиці (`tables/`)

| Файл | Колонки | Вимога |
|---|---|---|
| `data_state.csv` | вибірка (dev / повний train / test), рядків, позитивних, частка класу 1 | R1 |
| `baseline.csv` | параметри HGB ЛР2; PR-AUC 3-фолд (внутрішні розбиття) ± σ; PR-AUC 5-фолд (зовнішні) ± σ; ROC-AUC; train PR-AUC; розрив | R2 |
| `search_space.csv` | параметр, опис, значення сітки, нижня/верхня межа, шкала, тип | R5 |
| `optimization_methods.csv` | **Метод оптимізації**, **Кількість ітерацій / оцінок**, **Час виконання (сек)**, **Best Score (Inner CV)**, **Unbiased Score (Nested CV)** (значення «0,4812 ± 0,0190») — рядки: Baseline (ЛР2, без пошуку: «1 / 3», час навчання), Grid, RS20, RS50, Optuna | R18 |
| `optimization_cost.csv` | метод; навчань; процесів × потоків; wall, с; сумарний час навчань, с (між методами непорівнянний — різна кількість потоків); навчань/с; час Nested, с | R10, R21 |
| `optimization_quality.csv` | метод (+ рядок baseline); Best Score ± σ; Nested ± σ; оптимізм сирий (inner − outer) ± s; оптимізм відбору (мінус зсув baseline на тому самому фолді) ± s; приріст над baseline Δ ± s; Δ > s? | R17, R21 |
| `nested_pairwise.csv` | пари «метод A / метод B» (усі методи й baseline): Δ, s, \|Δ\| > s? | R17, R21 (правило D9) |
| `leakage_holdout.csv` | коректний і некоректний пайплайни: PR-AUC CV, PR-AUC test, CV − test, ROC-AUC test | R14 |
| `best_params.csv` | метод × 4 параметри (самостійний пошук) + baseline | R11 |
| `search_history.csv` | метод, номер оцінки, 4 параметри, mean / std PR-AUC, compute с, best-so-far | P5 |
| `nested_folds.csv` | метод (+ baseline), зовнішній фолд, inner best, outer PR-AUC, outer ROC-AUC, оптимізм, найкращі параметри, час пошуку | R15–R17 |
| `leakage_comparison.csv` | варіант, що витікає, PR-AUC CV ± σ, ROC-AUC CV, train PR-AUC, проти чого порівнюється, Δ ± s, Δ > s?, test PR-AUC (для двох головних), завищення CV − test | R12–R14, R19 |
| `leakage_folds.csv` | варіант × фолд: PR-AUC, ROC-AUC, train PR-AUC (для SMOTE ще naive-оцінки) | R14 |
| `leakage_by_size.csv` | обсяг, коректний ± σ, некоректний ± σ, Δ ± s | R14 |
| `test_results.csv` | модель (baseline, 4 методи), PR-AUC test, ROC-AUC test, Δ проти baseline, 95 % ДІ, «ДІ не містить 0» | P6 |

### 4.3. Ключі `metrics.json`

| Ключ | Зміст |
|---|---|
| `data` | `dev_rows`, `dev_positives`, `dev_minority_share`, `full_train_rows`, `test_rows`, `manifest_checked` |
| `metric` | `name` (`"pr_auc"`), `scoring` (`"average_precision"`), `secondary` (`"roc_auc"`) |
| `model` | `kind`, базові параметри, `lab2_train_cv_gap` |
| `search_space` | по параметру: `grid`, `low`, `high`, `log`, `integer`; `grid_size` |
| `baseline` | `inner_pr_auc`, `inner_pr_auc_std`, `outer_pr_auc`, `outer_pr_auc_std`, `outer_roc_auc`, `train_pr_auc`, `test_pr_auc`, `test_roc_auc`, `inner_in_outer_pr_auc`, `inner_minus_outer_mean`, `inner_minus_outer_std` (зсув inner − outer без відбору — 3-фолдова CV baseline у кожному зовнішньому train) |
| `methods.<grid\|random_20\|random_50\|optuna_20>` | `label`, `n_candidates`, `n_fits`, `wall_seconds`, `compute_seconds`, `evals_per_second`, `best_score_inner`, `best_score_inner_std`, `best_params`, `nested_mean`, `nested_std`, `nested_roc_auc`, `nested_inner_mean`, `optimism_raw_mean`, `optimism_raw_std`, `optimism_selection_mean`, `optimism_selection_std`, `optimism_selection_exceeds_spread`, `workers`, `threads_per_fit`, `best_minus_nested`, `gain_vs_baseline`, `gain_vs_baseline_std`, `nested_wall_seconds`, `test_pr_auc`, `test_roc_auc`, `test_delta_vs_baseline`, `test_delta_low`, `test_delta_high` |
| `methods_summary` | `time_basis` (`wall_seconds`), `fastest`, `slowest`, `wall_ratio_slowest_to_fastest`, `pairs_*` (усього / розрізнюваних: проти baseline і між методами), `baseline_shift_mean/std`. «Переможця» за якістю немає свідомо (D9) |
| `leakage.<variant>` | `pr_auc`, `pr_auc_std`, `roc_auc`, `train_pr_auc`, `delta`, `delta_std`, `delta_exceeds_spread`; для `leak_smote` ще `naive_pr_auc`, `naive_roc_auc` |
| `leakage.headline` | `correct_cv`, `incorrect_cv`, `inflation_cv` (Δ), `inflation_cv_std`, `correct_test`, `incorrect_test`, `inflation_vs_test` (CV − test некоректного), `test_delta`, `test_delta_low`, `test_delta_high` |
| `leakage.by_size.<n>` | `correct`, `incorrect`, `delta`, `delta_std` |
| `lab2_reference` | `hgb_full_test_pr_auc`, `hgb_full_cv_pr_auc` (якщо доступно) |
| `timing` | `total_seconds`, секунди кожного кроку, `n_cpu` |
| `config` | `seed`, `dev_sample_size`, фолди, `optuna_startup_trials`, `search_n_jobs`, `optuna_fold_n_jobs` |

## 5. Ризики та обмеження

### 5.1. Час обчислень (CPU, Apple M1 Pro)

Орієнтир: одне навчання пайплайна Б + HGB на 13,3 тис. рядків ≈ 2,9 с в одному потоці; при 10
процесах ≈ 0,33–0,37 с wall на навчання (проба під навантаженням; без нього, ймовірно, швидше).

| Частина | Навчань | Оцінка, хв |
|---|---|---|
| Baseline (3 + 5 фолдів) | 8 | < 0,2 |
| Етап 2: Grid / RS20 / RS50 / Optuna | 243 / 60 / 150 / 60 | ≈ 1,4 / 0,4 / 0,9 / 0,7 |
| Етап 4 (вкладена CV, ×5, трохи менші дані) | 1 215 / 300 / 750 / 300 + 20 refit | ≈ 6,5 / 1,6 / 4 / 3,5 |
| Етап 3 (витік, повний train і підвибірки, test) | ≈ 30 великих + 20 малих | ≈ 1,5–2 |
| Фінальні refit + test + графіки | 5 | < 0,5 |
| **Разом** | ≈ 3 150 | **≈ 18–22 хв** |

Ризики: (1) **стороннє навантаження** — під load ≈ 400 одне навчання HGB з 3 потоками OpenMP
тривало 54 с замість 2,5 с; головний запуск робити на вільній машині, у `metrics.timing` писати
`n_cpu` і загальний час; (2) wall-clock між запусками відрізняється на десятки відсотків — у звіті
порівнювати час методів **у межах одного запуску** (відношення), а не абсолютні секунди;
(3) якщо перший запуск > 25 хв — зменшити `dev_sample_size` до 15 000 (правило D4), а не скорочувати
сітку (81 — «повна» сітка 3⁴, природна для звіту).

### 5.2. Статистичні обмеження

- На 20 тис. σ PR-AUC зовнішнього фолда ≈ 0,02 (проба: коректний 5-фолд на 20 тис. — 0,464 ± 0,024).
  Різниці між методами, найімовірніше, менші за шум; оцінки «Best − Nested» по одному методу —
  того ж порядку. Звіт має чесно сказати «методи нерозрізнимі за якістю, різняться в рази за часом»,
  якщо так вийде. Парні різниці на тих самих зовнішніх фолдах зменшують шум, але 5 пар — мало.
- Best Score (Inner CV) **за побудовою** оптимістичний (максимум із 20–81 зашумлених оцінок);
  більше кандидатів → більший оптимізм. Очікувано Grid і RS50 матимуть найбільший оптимізм — це
  один з головних висновків, а не помилка.
- Grid у порядку `ParameterGrid` — крива збіжності сходинкова і залежить від порядку параметрів;
  пояснити, що порівнювати слід кінцеві точки.
- Optuna з 20 спробами, з яких 5 випадкових, — TPE ледве встигає «навчитися»; перевага над RS20
  може не проявитися.

### 5.3. Методичні обмеження

- **Test повторно використовується** після ЛР2 (D8). Жоден вибір у ЛР3 не робиться за test.
  **Проба плану торкнулася test** один раз (PR-AUC моделі з витоком на test, scratchpad) — це
  підтвердило, що перевірка «реальністю» показова, але жоден параметр, межа простору чи варіант
  витоку за цим числом не налаштовувався; у код і звіт воно не потрапляє.
- Межі простору (D3) звужено за пробою на **train-підвибірці** (не на test): `learning_rate=0.3`
  гірше на внутрішніх фолдах. Це легітимне «попереднє дослідження», але назвати у звіті.
- Модель, налаштована на 20 тис., не порівнюється напряму з HGB ЛР2 на 58 тис. (інший обсяг);
  `lab2_reference` — лише довідка.
- Варіант `leak_smote` (б) має іншу частку класів у валідації (33 % проти 12 %) — PR-AUC непорівнянна,
  порівнювати лише ROC-AUC; у звіті сказати прямо.
- `leak_unsupervised` не є «нульовим витоком»: `TargetEncoder` із крос-фітингом по всіх даних
  кодує val-рядок статистикою, у якій є мітки train-рядків і інших val-рядків (але не власна).
  Проба показала ефект у межах шуму — так і інтерпретувати.
- `max_features` у HGB використовує `random_state` — результати детерміновані при фіксованому seed;
  при `max_features=1.0` модель не залежить від зерна.

### 5.4. Дані та залежності

- Датасет: `data/processed/kick/` від ЛР1 (без ручних кроків; якщо його нема — `mllabs run 1`).
  ЛР3 залежить від ЛР2 лише через код `mllabs.modeling` і необов'язкове читання
  `labs/lab02/artifacts/metrics.json` (`lab2_reference`).
- Нових залежностей немає: `optuna` 5.0 уже в `pyproject.toml`; `threadpoolctl` — транзитивна
  залежність sklearn (якщо інженер імпортує її напряму, варто додати явно через `uv add threadpoolctl`
  — легка; або обійтися без неї, якщо заміри покажуть, що joblib сам обмежує потоки).
- `scikit-optimize` і `optuna-integration` не потрібні.

## Прийняті рішення (оркестратор, 2026-11-16)

Автор надав повну автономію; прийнято рекомендації плану:

- **D1** — HistGradientBoosting (параметри ЛР2 як baseline).
- **D4** — стратифікована підвибірка dev = 20 000 рядків; інженер калібрує за правилом D4
  (прогноз > 25 хв → 15 000, < 12 хв → 30 000) і записує виміряне обґрунтування в config.py.
- **D7** — п'ять пайплайнів (`correct`, `leak_unsupervised`, `leak_fit_all`, `correct_smote`,
  `leak_smote`) + розріз за обсягом + перевірка на test.
- Решта (D2, D3, D5, D6, D8, D9) — як рекомендовано.

Додатково:
- Високе навантаження під час проб (load 47–440) спричинили паралельні запуски самих агентів
  (рев'ю/реалізація ЛР2 і проби цього плану одночасно) та вкладений паралелізм joblib × OpenMP.
  Головний запуск ЛР3 робити, коли інші важкі запуски не йдуть; потоки OpenMP у воркерах
  обмежувати явно (threadpoolctl або OMP_NUM_THREADS), і перевірити це заміром.
- Проба планувальника торкнулася test (PR-AUC 0,466 для пайплайна з витоком). За цим числом
  нічого не налаштовано; у звіті не використовується.
- Межі простору звужено за пробою на train-підвибірці — у звіті назвати.
