# План лабораторної роботи №1

EDA, попередня обробка та аугментація даних. План складено за `ASSIGNMENT.md` ЛР1 з урахуванням
того, що ЛР2 (сценарії А/Б/В) і ЛР3 (витік даних, вкладена крос-валідація) використовуватимуть
той самий датасет і препроцесинг.

Усі цифри про датасети в цьому плані отримано пробним завантаженням 2026-11-13
(scikit-learn 1.9.1, pandas 3.1.1). Це орієнтири для інженера, **не** цифри для звіту: у звіт
числа потрапляють лише з `artifacts/`.

## 1. Вимоги (чекліст)

### 1.1. Вимоги із завдання

| ID | Вимога із завдання | Як буде виконано | Артефакт-доказ |
|---|---|---|---|
| R1 | Обрати складний публічний табличний датасет із Kaggle або UCI | Датасет за рішенням D1 (рекомендація — Carvana «Don't Get Kicked!», Kaggle; завантаження через дзеркало OpenML id 41162 без облікових даних). Кешується в `data/raw/` | `tables/dataset_overview.csv`, `metrics.dataset.*` |
| R2 | Не менше 10 ознак (числові + категоріальні) | Таблиця типів ознак після очищення: кількість числових і категоріальних, кардинальність кожної | `tables/feature_summary.csv`, `metrics.dataset.n_numeric`, `metrics.dataset.n_categorical` |
| R3 | Наявність дисбалансу класів (задача класифікації) | Частки класів цільової змінної, коефіцієнт дисбалансу, стовпчикова діаграма | `figures/target_distribution.png`, `metrics.dataset.minority_share`, `metrics.dataset.imbalance_ratio` |
| R4 | Обґрунтувати вибір датасету | У звіті: відповідність кожній вимозі (R1–R3) + «складність»: реальні пропуски, приховані пропуски (нулі в цінах), брудні категорії, висока кардинальність, мультиколінеарність | `tables/dataset_overview.csv`, `tables/missing_values.csv`, `figures/missing_values.png` |
| R5 | Побудувати матриці кореляцій Пірсона та Спірмена | Дві теплові карти числових ознак train-вибірки + таблиці коефіцієнтів | `figures/correlation_pearson.png`, `figures/correlation_spearman.png`, `tables/correlation_pearson.csv`, `tables/correlation_spearman.csv` |
| R6 | Виявити мультиколінеарність | Перелік пар з \|r\| вище порога (обидва коефіцієнти) + VIF кожної ознаки до і після відбору; рішення, які ознаки прибрати | `tables/high_correlation_pairs.csv`, `tables/vif.csv`, `metrics.correlation.*` |
| R7 | Перевірити числові ознаки на нормальність візуально | Гістограми з KDE та Q-Q графіки для всіх числових ознак | `figures/numeric_distributions.png`, `figures/qq_plots.png` |
| R8 | Перевірити нормальність статистичними тестами (Шапіро-Уїлка або Д'Агостіно) | Обидва тести: Шапіро-Уїлка на випадковій підвибірці 5000 (обмеження тесту), Д'Агостіно-Пірсона на всьому train; плюс асиметрія та ексцес | `tables/normality_tests.csv`, `metrics.normality.n_normal_features` |
| R9 | Порівняти підходи до виявлення викидів: IQR, Z-score та ансамблевий (Isolation Forest або LOF) | Чотири детектори (IQR, Z-score, Isolation Forest, LOF): частка викидів по ознаках і по рядках, попарний перетин (Жаккар), частка «поганих» авто серед викидів, 2D-проєкція | `tables/outliers_by_feature.csv`, `tables/outliers_by_method.csv`, `tables/outlier_overlap.csv`, `figures/outlier_overlap.png`, `figures/outliers_pca.png`, `metrics.outliers.*` |
| R10 | Порівняти базову імп'ютацію (медіана/мода) з просунутою (KNNImputer або IterativeImputer/MICE) | (а) Експеримент із маскуванням: у повних рядках штучно ховаємо частку значень і міряємо RMSE відновлення для median / KNN / MICE; (б) вплив на проксі-модель у крос-валідації; (в) категоріальні: мода проти окремої категорії «MISSING» | `tables/imputation_comparison.csv`, `figures/imputation_comparison.png`, `figures/imputation_distributions.png`, `metrics.imputation.*` |
| R11 | Target Encoding або Frequency Encoding для високопотужних категоріальних змінних | Обидва: `sklearn.preprocessing.TargetEncoder` (із внутрішнім крос-фітингом) і власний `FrequencyEncoder`; порівняння на проксі-моделі | `tables/encoding_comparison.csv`, `tables/cardinality.csv`, `figures/encoding_comparison.png`, `metrics.encoding.*` |
| R12 | One-Hot Encoding / Ordinal Encoding для інших ознак | One-Hot для номінальних ознак низької кардинальності; Ordinal — для бінарних ознак і як базове кодування сценарію «мінімальне очищення» | `tables/encoding_plan.csv` (ознака — кардинальність — метод — причина) |
| R13 | Масштабування: RobustScaler або PowerTransformer/Box-Cox для усунення скошеності | Порівняння «без перетворення / RobustScaler / PowerTransformer (Yeo-Johnson) / Box-Cox для строго додатних ознак»: асиметрія до і після, гістограми до і після | `tables/scaling_skewness.csv`, `figures/scaling_before_after.png`, `metrics.scaling.*` |
| R14 | Синтез нових спостережень: SMOTE, ADASYN або керований гаусів шум після розбиття | Три методи (SMOTE, ADASYN, гаусів шум для міноритарного класу), лише на train і лише всередині `imblearn.pipeline.Pipeline`; розподіл класів до/після, 2D-проєкція синтетичних точок, вплив на проксі-модель | `tables/augmentation_class_balance.csv`, `tables/augmentation_comparison.csv`, `figures/augmentation_class_balance.png`, `figures/augmentation_pca.png`, `metrics.augmentation.*` |
| R15 | Інженерія ознак: нові ознаки на основі взаємодії наявних (нелінійні комбінації, поліноміальні ознаки, агрегації за категоріями) | Три групи ознак: відношення/логарифми/частини дати; `PolynomialFeatures` 2-го степеня для кількох ознак; агрегації за категоріями (навчаються на train); оцінка корисності через взаємну інформацію та проксі-модель | `tables/engineered_features.csv`, `tables/feature_engineering_comparison.csv`, `figures/engineered_features_mi.png`, `metrics.feature_engineering.*` |
| R16 | Звіт: опис датасету та результати EDA з графіками й висновками | Код дає всі графіки/таблиці R1–R9; текст пише `report-writer` | артефакти R1–R9, `report/report.md` |
| R17 | Звіт: обґрунтування вибору методів імп'ютації, кодування та аугментації | Обґрунтування спирається на порівняльні таблиці R10, R11, R14 і підсумкову таблицю обраного препроцесингу | `tables/preprocessing_summary.csv`, артефакти R10, R11, R14 |
| R18 | Реалізація засобами Python і популярних бібліотек | scikit-learn, imbalanced-learn, pandas, scipy, seaborn; нових залежностей не потрібно | код у `src/mllabs/` |

### 1.2. Проєктні вимоги (не із завдання, з `CLAUDE.md` та залежностей ЛР2–3)

| ID | Вимога | Як буде виконано | Артефакт-доказ |
|---|---|---|---|
| P1 | Датасет і препроцесинг ЛР1 перевикористовуються в ЛР2–3 | Спільні пакети `mllabs/data/` і `mllabs/preprocessing/`; на диск передаються **очищені, але не трансформовані** train/test | `data/processed/<dataset>/train.csv`, `test.csv`, `schema.json` |
| P2 | Жодного `fit` на тестових даних | Розбиття одразу після детермінованого очищення; EDA, детектори, імп'ютери, енкодери, семплери — лише на train; порівняння методів — крос-валідацією всередині train. Тест у ЛР1 не використовується для жодної оцінки | тест `test_no_leakage_*`, `metrics.split.*` |
| P3 | Відтворюваність | `random_state=context.seed` у кожному розбитті, моделі, семплері, підвибірці | повторний запуск дає ті самі `metrics.json` |

## 2. Ключові рішення

### D1. Датасет — ПОГОДЖЕНО: варіант A (`kick`)

> Автор обрав варіант A (Carvana «Don't Get Kicked!») 2026-11-13. Рішення D2–D8 прийнято
> оркестратором за рекомендаціями цього плану.

Усі три варіанти реально завантажено кодом `fetch_openml(...)` без облікових даних.

| Характеристика | A. Carvana «Don't Get Kicked!» (`kick`) | B. Adult / Census Income | C. Speed Dating |
|---|---|---|---|
| Джерело | Kaggle (змагання Carvana), дзеркало OpenML **id 41162**, ліцензія CC0 | UCI, OpenML **id 1590** | OpenML **id 40536** (оригінал — Columbia, є на Kaggle) |
| Рядків | 72 983 | 48 842 | 8 378 |
| Ознак (без цілі) | 32: 14 числових + 18 категоріальних | 14: 6 числових + 8 категоріальних | 120: 59 числових + 61 категоріальна |
| Ціль, частка міноритарного класу | `IsBadBuy`, **12,3 %** (1 : 7,1) | `class` (>50K), 23,9 % (1 : 3,2) | `match`, 16,5 % (1 : 5,1) |
| Справжні пропуски в числових | так, але мало: NaN до 0,43 % (ціни MMR, `VehBCost`); ще ~1,1 % прихованих пропусків — нульові ціни MMR; разом ~1,7 % рядків | **немає** | багато: 35 ознак із пропусками >1 %, до 78,5 % |
| Справжні пропуски в категоріальних | `Trim` 3,2 %, `WheelType` 4,3 %, `PRIMEUNIT`/`AUCGUART` 95,3 % | `workclass` 5,7 %, `occupation` 5,8 %, `native-country` 1,8 % | практично немає |
| Висока кардинальність | **`Model` 1063, `SubModel` 863, `VNZIP1` 153, `Trim` 134, `BYRNO` 74**, `VNST` 37, `Make` 33 | лише `native-country` 41 | лише `field` 259 |
| Мультиколінеарність | яскрава: 8 цін MMR з r = 0,91–0,99; `VehYear`/`VehicleAge` r = −0,96 | слабка (`education`/`education-num` дублюються) | є (дубльовані «бінові» ознаки `d_*`) |
| Скошеність | помірна (`WarrantyCost` 2,07), важкі хвости в цінах | сильна (`capital-gain` 11,9) | помірна |
| Особливості | брудні дані: `Manual`/`MANUAL`, `NOT AVAIL`, нулі замість цін; пропуск у `WheelType` інформативний (частка «поганих» авто 70 % проти 12 %); є дата покупки | «підручниковий», добре відомий викладачам; мало матеріалу для імп'ютації числових і для Target Encoding | містить ознаки-витоки (`decision`, `decision_o` визначають `match`) і потребує ручного відбору з 120 колонок |
| Завантаження | `fetch_openml(data_id=41162, parser="liac-arff")`; ARFF 16 МБ, ~2 с | `fetch_openml(data_id=1590)`; ~1,5 с | `fetch_openml(data_id=40536, parser="liac-arff")` |
| Час повного запуску ЛР1 | ~1–2 хв | <1 хв | <1 хв |

**Рекомендація: варіант A (`kick`).** Єдиний, що буквально закриває всі пункти завдання одночасно:
походить із Kaggle, 32 ознаки обох типів, виражений дисбаланс 12,3 %, справжні пропуски обох
типів, п'ять ознак високої кардинальності (Target/Frequency Encoding матимуть сенс, а не будуть
формальністю), сильна мультиколінеарність для R6. Для ЛР3 висока кардинальність дає наочний
витік даних (Target Encoding до розбиття), для ЛР2 12 % дисбалансу — змістовний сценарій В.

Слабке місце A: справжніх пропусків у числових ознаках мало (~1,7 % рядків), тому різниця між
медіаною і KNN/MICE на реальних пропусках буде майже непомітною. Це закривається експериментом із
маскуванням (R10): на пробі RMSE відновлення медіана 0,72 проти KNN 0,31 і MICE 0,30 — порівняння
виходить змістовним. Якщо автору важливіше мати багато справжніх числових пропусків — варіант C,
але він вимагає ручного відбору ознак і удвічі більше пояснень у звіті.

Відхилено після перевірки: Diabetes130US (id 4541; пропуски закодовані як категорія, 101 тис.
рядків, 3 класи), `sick` (id 38; є числові пропуски до 20 %, але немає високої кардинальності,
максимум 5 категорій), Bank Marketing (id 1461; пропусків немає), weatherAUS / HR Analytics /
Stroke (на OpenML не знайдено, лише Kaggle з обліковими даними).

Подальші розділи написано для варіанта A. Для B або C змінюються лише `schema`, правила
`RawDataCleaner` і перелік інженерних ознак; структура коду та артефактів та сама.

### D2. Що передається в ЛР2–3

| Варіант | Плюси | Мінуси |
|---|---|---|
| 1. Готова трансформована матриця (після імп'ютації, кодування, SMOTE) | просто прочитати | унеможливлює сценарії А/Б/В і чесну крос-валідацію; сам по собі є витоком даних |
| 2. **Очищені нетрансформовані train/test + фабрики нефітованих пайплайнів** | ЛР2 збирає три сценарії з однієї фабрики; ЛР3 будує «коректний» і «некоректний» пайплайн з тих самих цеглин | трохи більше коду в спільному пакеті |

**Рекомендація: варіант 2.** «Підготовлений у ЛР1 датасет» = файли `data/processed/kick/` +
`PreprocessorFactory`. Сценарії: А — `PreprocessingScenario.MINIMAL` (медіана/мода, Ordinal, без
масштабування); Б — `FULL` (обрана в ЛР1 імп'ютація + Target/One-Hot + PowerTransformer);
В — `FULL` + семплер із `SamplerFactory` в `imblearn.pipeline.Pipeline`.

### D3. Порядок «розбиття — EDA»

| Варіант | Плюси | Мінуси |
|---|---|---|
| 1. EDA на всьому датасеті, розбиття потім | більше даних для статистик | рішення препроцесингу приймаються з підгляданням у тест |
| 2. **Опис датасету на всіх даних, розбиття, далі все на train** | методично чисто; узгоджується з темою ЛР3 | — |

**Рекомендація: варіант 2.** На повних даних рахується лише опис (розмір, типи, пропуски, баланс
класів). Розбиття стратифіковане 80/20. У даних є дата покупки, тож можливе і розбиття за часом;
обрано випадкове стратифіковане, бо цього вимагає `CLAUDE.md`, а ЛР3 передбачає `StratifiedKFold`.

### D4. Як порівнювати методи (імп'ютація, кодування, аугментація, нові ознаки)

| Варіант | Плюси | Мінуси |
|---|---|---|
| 1. Лише описово (розподіли до/після) | швидко | «обґрунтування вибору» (R17) без чисел |
| 2. **Проксі-модель у крос-валідації на train** | дає числове обґрунтування; тест лишається недоторканим для ЛР2–3 | ЛР1 частково торкається моделювання |
| 3. Кілька моделей і тестова вибірка | повніше | дублює ЛР2, «спалює» тест |

**Рекомендація: варіант 2.** Одна проста модель — `LogisticRegression(max_iter=500)`,
`StratifiedKFold(3)`, головна метрика **PR-AUC** (`average_precision`), додатково ROC-AUC, F1 і
recall міноритарного класу (accuracy при 12 % дисбалансу неінформативна). Модель — вимірювальний
інструмент, а не результат роботи; порівняння алгоритмів лишається для ЛР2.
Попередження: на пробі різниця між кодуваннями мала (PR-AUC 0,42–0,44), а SMOTE PR-AUC не
підвищив (0,421 проти 0,431). Це нормальний результат — висновки пишуться за фактичними числами.

### D5. Що робити з викидами

| Варіант | Плюси | Мінуси |
|---|---|---|
| 1. Видалити рядки-викиди з train | чистіші розподіли | тестові рядки видаляти не можна; хвости цін — справжні дорогі авто; втрата міноритарного класу |
| 2. **Не видаляти; пом'якшити PowerTransformer / RobustScaler** | однаково діє на train і test, живе в Pipeline | викиди лишаються в даних |
| 3. Вінзоризація (обрізання до квантилів) | проста | ще один гіперпараметр |

**Рекомендація: варіант 2.** Детектори використовуються для аналізу (R9), а не для фільтрації.
Очевидно помилкові значення (ціна 0 або 1 долар) обробляються раніше як приховані пропуски.

### D6. Мультиколінеарність

Варіанти: (1) лишити все; (2) **прибрати дублікати за переліком у конфігурації**; (3) PCA для цін.
**Рекомендація: (2)** — у сценарії `FULL` прибрати `VehYear` (дублює `VehicleAge`) і чотири
`*CleanPrice` (r = 0,99 з відповідними `*AveragePrice`); решту цін MMR замінюють/доповнюють
відношення з R15. Рішення підтверджується таблицею VIF «до/після». PCA відхилено: втрачається
інтерпретованість, а для дерев у ЛР2 вона не потрібна. У сценарії `MINIMAL` нічого не прибирається.

*Після повторного рев'ю:* усі рішення `FULL` ухвалюються за єдиним правилом вибору
(`mllabs.preprocessing.selection`): якщо |Δ PR-AUC| ≤ σ між фолдами, якість однакова й обирається
простіший варіант (менше ознак, нижчий VIF, швидший метод); складніший — лише якщо він кращий більше
ніж на σ. Правило застосовується до ОСТАТОЧНОЇ конфігурації `FULL` (без агрегацій): для неї
залишення `AcqRetAvg` і `CurRetAvg` дає виграш PR-AUC більший за σ варіанта без них, тому ці ціни
лишаються в `FULL` (додаткове вилучення лише перевіряється, VIF > 10 у них прийнятний за
L2-регуляризації). Єдиний свідомий виняток — Target Encoding при кращому One-Hot (вимога
завдання й у 10 разів менше ознак).

*Після рев'ю наставника:* правило вибору замінено на парне порівняння по фолдах: обидва
варіанти оцінюються на тих самих 5 фолдах `StratifiedKFold` (`proxy_cv_folds` 3 → 5),
d_k = PR-AUC складнішого − PR-AUC простішого, Δ = mean(d), s = std(d, ddof=1); складніший
обирається, лише якщо Δ > s. Кожне рішення перевіряється відносно остаточного `FULL`. За новим
правилом роздрібні ціни `AcqRetAvg`, `CurRetAvg` лишаються (Δ > s), агрегації за категоріями
УВІМКНЕНО в `FULL` (Δ > s), поліном вимкнено (Δ ≤ s). Категоріальні пропуски — окрема
категорія MISSING, і це ДРУГИЙ свідомий виняток із правила (поруч із Target Encoding): MISSING
складніший (92 проти 88 ознак — колонки `_MISSING` також у Color, Size, Nationality,
TopThreeAmericanName), Δ ≤ s, тож правило радить «моду + індикатор». Але тоді інформація про
невипадковий пропуск WheelType (найсильніший предиктор) існує лише в індикаторі з групи
комбінацій, і будь-яка конфігурація без цієї групи (база R14/R15, сценарії ЛР2–3) її втрачає
(PR-AUC бази ≈ 0,43 → 0,25). Винятки оголошено в коді (`_DECLARED_EXCEPTIONS` у `summary.py`);
Δ, s і висновок кожного рішення — у `metrics.selection`.

### D7. Колонки з 95 % пропусків (`PRIMEUNIT`, `AUCGUART`) і дублікат `WheelTypeID`

**Рекомендація:** `WheelTypeID` видаляє `RawDataCleaner` (той самий зміст, що `WheelType`).
`PRIMEUNIT`/`AUCGUART` показуються в таблиці пропусків і видаляються за порогом
`max_missing_share = 0.5` — заповнювати модою 95 % значень безглуздо, і це хороший приклад для
звіту. Пропуски в `WheelType` і `Trim` заповнюються окремою категорією `MISSING` (пропуск
інформативний), мода — як базовий варіант для порівняння.

### D8. Формат файлів у `data/processed/`

Варіанти: (1) **CSV + `schema.json` з типами**, без нових залежностей; (2) Parquet — зберігає типи
сам, але потребує `uv add pyarrow` (у середовищі зараз немає). **Рекомендація: (1).** Важливо:
`VNZIP1`, `BYRNO` виглядають як числа, тому читати CSV слід із явними типами зі схеми.

## 3. Дизайн коду

### 3.1. Спільні пакети (використовують ЛР1–3)

`src/mllabs/data/`

| Файл | Клас / функція | Відповідальність |
|---|---|---|
| `schema.py` | `DatasetSchema` (`@dataclass(frozen=True)`), константа `KICK_SCHEMA` | Імена: ціль, позитивний клас, числові ознаки, категоріальні низької та високої кардинальності, бінарні, колонка дати, колонки на видалення |
| `sources.py` | `DatasetSource` (`Protocol`), `OpenMlDatasetSource` | Завантаження сирої таблиці: `fetch_openml(data_id, parser="liac-arff", data_home=project.data_raw_dir / "openml")`; повторні запуски беруть кеш |
| `cleaning.py` | `RawDataCleaner` | Детерміновані правила без навчання: типи; ціль у `int`; `PurchDate` (unix-секунди) у дату; `Manual` → `MANUAL`; `NOT AVAIL` → пропуск; ціни MMR ≤ 1 → пропуск; видалення `WheelTypeID`; звіт про кількість змін кожним правилом |
| `splitting.py` | `DatasetSplit` (frozen-датаклас: `x_train`, `x_test`, `y_train`, `y_test`), `StratifiedSplitter` | Стратифіковане розбиття з `random_state` |
| `repository.py` | `ProcessedDatasetRepository` | `save(split, schema)` / `load() -> DatasetSplit` у `data/processed/kick/`; читання CSV з типами зі схеми; зрозумілий виняток, якщо ЛР1 ще не запускали |

`src/mllabs/preprocessing/`

| Файл | Клас / функція | Відповідальність |
|---|---|---|
| `encoders.py` | `FrequencyEncoder` (`BaseEstimator`, `TransformerMixin`) | Заміна категорії її часткою в train; невідомі категорії → 0 |
| `features.py` | `RowFeatureEngineer` (без стану), `GroupAggregateEncoder` (зі станом) | Відношення/логарифми/частини дати; агрегати за категорією, навчені на train, і відхилення від них |
| `augmentation.py` | `GaussianNoiseSampler` (нащадок `imblearn.base.BaseSampler`), `SamplerKind` (`StrEnum`), `SamplerFactory` | Копії міноритарного класу з шумом `N(0, (noise_scale * std)^2)`; фабрика SMOTE / ADASYN / шум. Семплер діє лише у `fit`, тому тест не аугментується |
| `factory.py` | `PreprocessingScenario` (`StrEnum`: `MINIMAL`, `FULL`), `PreprocessingOptions` (frozen-датаклас), `PreprocessorFactory` | `build(scenario) -> ColumnTransformer` — **нефітований**; параметри: вид імп'ютера, вид кодування високої кардинальності, вид масштабування, чи додавати інженерні ознаки, які колінеарні ознаки прибрати |
| `evaluation.py` | `ProxyModelEvaluator`, `CvScores` (frozen-датаклас) | Збирає `imblearn.pipeline.Pipeline(препроцесор [+ семплер] + LogisticRegression)` і повертає середні метрики крос-валідації на train |

### 3.2. Пакет лабораторної `src/mllabs/labs/lab01/`

| Файл | Клас | Відповідальність |
|---|---|---|
| `lab.py` | `Lab1(Lab)` | Лише перелік `Step` і короткі методи «взяти дані — викликати клас — зберегти через `ArtifactStore`» |
| `config.py` | `Lab1Config` | Усі числа (п. 3.4) |
| `overview.py` | `DatasetProfiler` | Розмір, типи, кардинальність, пропуски, баланс класів |
| `correlation.py` | `CorrelationAnalyzer` | Матриці Пірсона/Спірмена, пари вище порога, VIF (через обернену кореляційну матрицю, без `statsmodels`) |
| `normality.py` | `NormalityTester` | Шапіро-Уїлка (підвибірка), Д'Агостіно, асиметрія, ексцес |
| `outliers.py` | `OutlierDetector` (`Protocol`), `IqrDetector`, `ZScoreDetector`, `IsolationForestDetector`, `LofDetector`, `OutlierComparison` | Кожен детектор повертає булеву маску рядків; порівняння — частки, перетин Жаккара, частка позитивного класу серед викидів |
| `imputation.py` | `ImputationBenchmark` | Маскування MCAR на повних рядках, RMSE по методах і ознаках |
| `encoding.py` | `EncodingBenchmark` | План кодування за кардинальністю; порівняння Target / Frequency / One-Hot з `min_frequency` / Ordinal на проксі-моделі |
| `scaling.py` | `ScalingBenchmark` | Асиметрія до/після для кожного перетворення |
| `augmentation.py` | `AugmentationBenchmark` | Баланс класів до/після, координати PCA реальних і синтетичних точок, метрики проксі-моделі |
| `feature_engineering.py` | `FeatureEngineeringBenchmark` | Перелік нових ознак з формулами, взаємна інформація з ціллю, метрики проксі-моделі з/без нових ознак |
| `plots.py` | функції `plot_*(...) -> Figure` | Усі графіки; нічого не рахують і не зберігають |

Тести: `tests/data/`, `tests/preprocessing/`, `tests/labs/test_lab01_<модуль>.py` — на крихітних
синтетичних таблицях і без мережі (фальшивий `DatasetSource`). Обов'язкові: детектор IQR на
відомих межах; `FrequencyEncoder` і `GroupAggregateEncoder` не бачать test; `GaussianNoiseSampler`
не змінює реальні рядки й дає заданий баланс; `RawDataCleaner` перетворює нульові ціни на пропуски;
`ProcessedDatasetRepository` зберігає типи при записі-читанні.

### 3.3. Кроки `Step` у порядку виконання

1. «1.1 Завантаження датасету» — `OpenMlDatasetSource` → сира таблиця.
2. «1.2 Очищення та опис датасету» — `RawDataCleaner`, `DatasetProfiler` (R1–R4).
3. «1.3 Стратифіковане розбиття» — `StratifiedSplitter`; далі все на train (P2).
4. «2.1 Кореляції та мультиколінеарність» (R5, R6).
5. «2.2 Перевірка нормальності» (R7, R8).
6. «2.3 Виявлення викидів» (R9).
7. «3.1 Імп'ютація пропусків» (R10).
8. «3.2 Кодування категоріальних ознак» (R11, R12).
9. «3.3 Масштабування та степеневі перетворення» (R13).
10. «4.1 Аугментація» (R14).
11. «4.2 Інженерія ознак» (R15).
12. «5. Підсумок препроцесингу та збереження даних для ЛР2–3» — `tables/preprocessing_summary.csv`,
    `ProcessedDatasetRepository.save` (R17, P1).

### 3.4. Параметри `Lab1Config`

| Параметр | Значення | Чому |
|---|---|---|
| `openml_data_id` | 41162 | датасет `kick`, версія 1 |
| `test_size` | 0.2 | ~14,6 тис. тестових рядків, ~1,8 тис. позитивних |
| `max_missing_share` | 0.5 | колонки з більшою часткою пропусків видаляються |
| `high_cardinality_threshold` | 30 | від цієї кількості категорій — Target/Frequency Encoding |
| `correlation_threshold` | 0.9 | поріг «сильної» кореляції для таблиці пар |
| `vif_threshold` | 10.0 | загальноприйнята межа мультиколінеарності |
| `normality_alpha` | 0.05 | рівень значущості тестів |
| `shapiro_sample_size` | 5000 | p-значення Шапіро-Уїлка в scipy ненадійне для N > 5000 |
| `iqr_factor` | 1.5 | правило Тьюкі |
| `zscore_threshold` | 3.0 | правило трьох сигм |
| `outlier_contamination` | 0.03 | однакова очікувана частка для Isolation Forest і LOF, щоб порівняння було чесним (із `"auto"` Isolation Forest позначає ~14 % рядків, LOF ~2 %) |
| `isolation_forest_estimators` | 200 | стабільніші оцінки, час <1 с |
| `lof_neighbors` | 20 | стандартне значення sklearn |
| `imputation_sample_size` | 10000 | KNNImputer квадратичний за кількістю рядків; на 10 тис. — ~3 с |
| `imputation_mask_share` | 0.1 | частка штучно прихованих значень |
| `knn_imputer_neighbors` | 5 | стандартне значення |
| `mice_max_iter` | 25 | *після рев'ю:* із 10 MICE у пайплайні не сходився (`ConvergenceWarning`), сходиться за ~21 ітерацію |
| `onehot_min_frequency` | 50 | рідкісні категорії об'єднуються, інакше One-Hot для `Model` дає понад 1000 колонок |
| `proxy_cv_folds` | 3 | достатньо для порівняння, утричі дешевше за 10 |
| `proxy_max_iter` | 500 | збіжність логістичної регресії |
| `smote_neighbors` | 5 | стандартне значення SMOTE/ADASYN |
| `sampling_strategy` | 0.5 | міноритарний клас доводиться до 50 % від мажоритарного; повне вирівнювання 1:1 створює ~45 тис. синтетичних рядків і підвищує ризик перенавчання (тема ЛР2) |
| `noise_scale` | 0.05 | шум 5 % стандартного відхилення ознаки |
| `polynomial_degree` | 2 | взаємодії та квадрати |
| `polynomial_features` | `VehicleAge`, `VehOdo`, `VehBCost` | поліном лише для кількох ознак, щоб не роздувати матрицю |
| `pca_plot_sample_size` | 3000 | точок на діаграмах розсіювання |

Перелік колінеарних ознак на видалення та склад груп ознак — у `KICK_SCHEMA`, не в конфігурації.

### 3.5. Інженерні ознаки (R15)

- Нелінійні комбінації: `cost_to_mmr = VehBCost / MMRAcquisitionAuctionAveragePrice`;
  `mmr_trend = MMRCurrentAuctionAveragePrice / MMRAcquisitionAuctionAveragePrice`;
  `miles_per_year = VehOdo / (VehicleAge + 1)`; `warranty_share = WarrantyCost / VehBCost`.
  *Після рев'ю:* `retail_margin` (точна лінійна комбінація двох цін) і `log_warranty`
  (після Yeo-Johnson дублює `WarrantyCost`, VIF ≈ 7·10⁵) прибрано; відношення
  `AcqRetAvg / AcqAucAvg` перевірено — PR-AUC не покращує, VIF числового блоку зростає до 640.
- Частини дати з `PurchDate`: рік, місяць, день тижня (сам timestamp у модель не йде).
- Індикатори пропусків: `wheel_type_missing`, `mmr_missing`. *Після рев'ю:* у `FULL`
  `wheel_type_missing` не подається — він дослівно дорівнює One-Hot колонці `WheelType_MISSING`.
- Поліноміальні: `PolynomialFeatures(degree=2, include_bias=False)` для `polynomial_features`.
- Агрегації за категоріями (fit на train): середня `VehBCost` за `Model` і `Make`, медіанний
  `VehOdo` за `VehicleAge`; відхилення авто від середнього своєї групи. *Після рев'ю:* у
  модель іде лише відхилення (статистика групи + ознака + відхилення — точна лінійна залежність).
  *Після повторного рев'ю:* агрегації в `FULL` не входять (`add_group_aggregates=False`): їхній
  внесок у PR-AUC у межах σ, а VIF вони підвищують. У бенчмарку R15 вони, як і поліном, і далі
  будуються та оцінюються.
  *Після рев'ю наставника:* за парним правилом (Δ > s) агрегації в `FULL` знову УВІМКНЕНО
  (лише відхилення від групи); `wheel_type_missing` у `FULL` не подається (дублює One-Hot
  `WheelType_MISSING`), `mmr_missing` лишається в групі комбінацій. Агрегації **не**
  використовують цільову змінну (інакше це було б Target Encoding).

## 4. Артефакти

### Графіки (`labs/lab01/artifacts/figures/`)

`target_distribution.png`, `missing_values.png`, `correlation_pearson.png`,
`correlation_spearman.png`, `numeric_distributions.png`, `qq_plots.png`, `outlier_overlap.png`,
`outliers_pca.png`, `imputation_comparison.png`, `imputation_distributions.png`,
`encoding_comparison.png`, `scaling_before_after.png`, `augmentation_class_balance.png`,
`augmentation_pca.png`, `engineered_features_mi.png` — 15 файлів.

### Таблиці (`labs/lab01/artifacts/tables/`)

`dataset_overview.csv`, `feature_summary.csv`, `missing_values.csv`, `cleaning_log.csv`,
`correlation_pearson.csv`, `correlation_spearman.csv`, `high_correlation_pairs.csv`, `vif.csv`,
`normality_tests.csv`, `outliers_by_feature.csv`, `outliers_by_method.csv`, `outlier_overlap.csv`,
`imputation_comparison.csv`, `cardinality.csv`, `encoding_plan.csv`, `encoding_comparison.csv`,
`scaling_skewness.csv`, `augmentation_class_balance.csv`, `augmentation_comparison.csv`,
`engineered_features.csv`, `feature_engineering_comparison.csv`, `preprocessing_summary.csv` —
22 файли. Великі матриці кореляцій у звіт ідуть графіком, а CSV лишаються як доказ.

### Ключі `metrics.json`

`ArtifactStore.save_metrics` зливає лише верхній рівень словника, тому кожен крок пише один
власний ключ верхнього рівня цілком.

| Ключ | Вкладені поля |
|---|---|
| `dataset` | `name`, `openml_id`, `n_rows`, `n_features`, `n_numeric`, `n_categorical`, `n_high_cardinality`, `max_cardinality`, `minority_share`, `imbalance_ratio`, `missing_cells_share`, `rows_with_missing_share`, `hidden_missing_cells` |
| `split` | `train_rows`, `test_rows`, `train_minority_share`, `test_minority_share` |
| `correlation` | `n_pairs_above_threshold`, `max_abs_pearson`, `max_abs_spearman`, `max_vif_before`, `max_vif_after` |
| `normality` | `n_features_tested`, `n_normal_features`, `max_abs_skew`, `most_skewed_feature` |
| `outliers` | `iqr_share`, `zscore_share`, `isolation_forest_share`, `lof_share`, `all_methods_share`, `positive_rate_in_outliers` (по методах) |
| `imputation` | `rmse_median`, `rmse_knn`, `rmse_mice`, `best_method`, `pr_auc` (по методах), `wheel_type_missing_positive_rate` |
| `encoding` | `pr_auc` і `n_output_features` по методах (`target`, `frequency`, `onehot`, `ordinal`), `best_method` |
| `scaling` | `mean_abs_skew_raw`, `mean_abs_skew_robust`, `mean_abs_skew_power`, `chosen_method` |
| `augmentation` | для `none`, `smote`, `adasyn`, `gaussian_noise`: `train_rows`, `minority_share`, `pr_auc`, `roc_auc`, `f1`, `recall` |
| `feature_engineering` | `n_new_features`, `pr_auc_before`, `pr_auc_after`, `top_feature_by_mi` |

### Дані для наступних лабораторних (`data/processed/kick/`)

`train.csv`, `test.csv` (очищені, нетрансформовані, з цільовою колонкою), `schema.json`.

## 5. Ризики та обмеження

- **Час.** Проба на M1 Pro для варіанта A: усі обчислення разом ~40 с (найдовші — крос-валідація
  One-Hot ~13 с, KNNImputer ~5 с, LOF ~3 с); з побудовою 15 графіків при 200 dpi повний
  `mllabs run 1` оцінюється в 1–2 хв. Перший запуск додатково качає 16 МБ. TensorFlow не потрібен.
- **Завантаження.** `fetch_openml` для id 41162 із `parser="auto"` або `"pandas"` падає з
  `TypeError: expected string or bytes-like object, got 'int'` (sklearn 1.9.1 + pandas 3.1.1);
  працює лише `parser="liac-arff"`. Резерв, якщо зламається API OpenML: пряме посилання
  `https://openml.org/data/v1/download/19335679/kick.arff` (перевірено, 200 OK).
- **Типи після завантаження.** Номінальні колонки приходять як `category` з рядковими значеннями,
  зокрема ціль (`'0'`/`'1'`), `VNZIP1`, `BYRNO`, `IsOnlineSale`; `PurchDate` — unix-секунди у
  `float`. Усе це приводить `RawDataCleaner`.
- **Розмір для ЛР2–3.** 58 тис. рядків train — забагато для SVM з RBF-ядром і для вкладеної
  крос-валідації з повним Grid Search. Розмір не обмежуємо в ЛР1: `ProcessedDatasetRepository.load`
  варто одразу зробити з необов'язковим параметром стратифікованої підвибірки, а рішення про її
  розмір приймати в планах ЛР2 і ЛР3.
- **KNNImputer** квадратичний за числом рядків із пропусками: на реальних даних швидкий (пропусків
  мало), але в експерименті з маскуванням обов'язкова підвибірка `imputation_sample_size`.
- **Маскування MCAR** моделює випадкові пропуски, тоді як реальні пропуски тут невипадкові
  (`WheelType`). Це обмеження слід прямо назвати у звіті.
- **Target Encoding** для `Model` (556 з 1063 категорій мають менше 10 рядків) схильний до
  перенавчання; використовувати лише `sklearn.preprocessing.TargetEncoder` усередині Pipeline
  (він робить крос-фітинг у `fit_transform`), не рахувати середні вручну.
- **Box-Cox** вимагає строго додатних значень: застосовується лише до таких ознак після
  імп'ютації; основний метод — Yeo-Johnson.
- **SMOTE після кодування.** SMOTE інтерполює між точками, тому One-Hot колонки синтетичних рядків
  стають дробовими. Для логістичної регресії це прийнятно; альтернатива `SMOTENC` значно повільніша
  на ознаках високої кардинальності. Обмеження назвати у звіті.
- **Застарілі ключі.** `metrics.json` між запусками не очищується; при перейменуванні ключів старі
  лишаються у файлі. Перевірка «запуск з нуля» — з видаленим каталогом `artifacts/`.
- **Некрасиві результати ймовірні**: аугментація може не підвищити PR-AUC, а просунута імп'ютація
  майже не вплинути на модель через малу частку пропусків. Висновки пишуться за фактом.
