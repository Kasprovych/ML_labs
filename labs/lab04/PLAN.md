# План лабораторної роботи №4

Детекція аномалій на зображеннях без учителя: патчі → ознаки переднавчених backbone (Keras) →
власна модель **Feature Autoencoder** (відновлення ознак патчів), дослідження регуляризації,
розміру патчів і екстракторів, профілювання FLOPs / latency, порівняння з PaDIM і PatchCore.
Датасет — **VisA** (новий; ЛР1–3 працювали з табличним `kick`). Модель і протокол спроєктовано
так, щоб ЛР5 (Integrated Gradients, Grad-CAM, perturbation analysis з reconstruction error,
FP/FN, edge-рекомендації) працювала з нею без переробок.

Перевірки, на які спирається план (2026-11-16, Apple M1 Pro, машина **сильно завантажена**
запуском ЛР3, load average 100–120). Цифри часу нижче — орієнтири для інженера, не для звіту:

- Дані VisA (перевірено на capsules): зображення 1500×1000 JPG; маски PNG режиму `L` зі значеннями
  0…5 (індекси дефектів, бінаризувати `> 0`); частка площі дефекту — медіана 0,15 %, p10 0,014 %,
  p90 1,65 % → сильна піксельна незбалансованість; після зменшення маски до 256×256 (nearest)
  порожніх масок 0 зі 100; декодування + resize ≈ 12 мс/зобр.
- `https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar` — HTTP 200,
  1 929 840 640 байт, без облікових даних; швидкість завантаження в пробі ≈ 0,5–1 МБ/с.
- Офіційне розбиття `split_csv/1cls.csv` (є у репозиторії `amazon-science/spot-diff`): кількості
  в розд. D2 узято з нього.
- Автоматичне посилання Anomalib на MVTec AD (`mydrive.ch/...`) — HTTP 404; сам Anomalib
  відключив автозавантаження MVTec AD.
- `tensorflow-metal 1.2.0` + TF 2.21 в окремому venv: `import tensorflow` **падає**
  (`libmetal_plugin.dylib: Library not loaded: _pywrap_tensorflow_internal.so`) — GPU для TF немає.
- `anomalib 2.7.0` + `torch 2.15.0` + `torchvision 0.29.2` встановлюються на macOS arm64 / Py 3.12
  (≈ 1 хв, 1,1 ГБ venv), `from anomalib.models import Padim, Patchcore` працює, MPS доступний;
  спільне розв'язання залежностей з TF 2.21 / numpy 2.5.4 / sklearn 1.9.1 проходить (`uv pip compile`).
  Навчання Anomalib **не** запускалося.
- Підрахунок FLOPs у TF 2.21 через `tensorflow.python.profiler.model_analyzer.profile` на
  «замороженому» графі (`convert_variables_to_constants_v2`) працює для Keras 3 моделей.
  Обрізані backbone при вході 256×256: ResNet50 (до `conv4_block6_out`) ≈ 4,6 GFLOPs,
  MobileNetV2 (до `block_13_expand_relu`) ≈ 0,32 GFLOPs, EfficientNetB0 (до `block6a_…`) ≈ 0,35
  GFLOPs (усі числа — для іншої точки обрізання в пробі, інженер перерахує на фінальних шарах).
- Головка-автоенкодер 512→256→64→256→512 (296 тис. параметрів): ≈ 9 с на епоху зі 100 тис.
  векторів при 4 потоках **під навантаженням** (без навантаження очікувано 2–3 с).
- Жадібний coreset PatchCore у TF: 12,8 мс на ітерацію при N = 200 тис. × 128 під навантаженням →
  для N ≈ 780 тис. і 1 % ≈ 6–7 хв під навантаженням, ≈ 1,5–2 хв без нього.

## 1. Вимоги (чекліст)

### 1.1. Вимоги із завдання

| ID | Вимога із завдання | Як буде виконано | Артефакт-доказ |
|---|---|---|---|
| R1 | Обрати складний датасет із високою дисперсією тла або незбалансованістю цілей | VisA, 3 категорії (D1, D2): `macaroni2` (неоднорідне текстуроване тло, найважча категорія VisA), `capsules` (кілька об'єктів у випадкових позиціях), `pcb4` (складна структура, 7 типів дефектів). Обидві ознаки складності **вимірюються**: дисперсія тла — попіксельне σ між нормальними train-зображеннями та σ яскравості фону; незбалансованість — частка аномальних пікселів і площа дефекту на зображенні (у capsules медіана площі дефекту ≈ 0,15 % зображення) | `tables/data_overview.csv`, `tables/background_variance.csv`, `tables/defect_area.csv`, `figures/samples_grid.png`, `figures/background_variance.png`, `figures/defect_area_hist.png`, `metrics.data.*` |
| R2 | Розбиття зображень на патчі | Два рівні (D4): (а) кожне зображення 256×256 → сітка 32×32 патчів; ознака патча — вектор backbone на кроці 8 з локальною агрегацією p×p (як у PatchCore); (б) явне розбиття зображення на тайли 2×2 (кожен тайл → 256×256) у дослідженні розміру патчів | `figures/patch_grid.png` (зображення з накладеною сіткою патчів і тайлів), `tables/transformations.csv`, `metrics.patches.*` |
| R3 | Вилучити ознаки переднавченими backbone (ResNet, MobileNet) | `keras.applications` з вагами ImageNet: ResNet50, MobileNetV2, EfficientNetB0; два проміжні шари (крок 8 і 16) → апсемплінг до 32×32 → конкатенація → усереднення каналів до 512 (D5). Препроцесинг кожного backbone — всередині моделі. Ознаки кешуються на диск (float16) | `tables/backbones.csv` (шари, канали, параметри, GFLOPs), `metrics.features.*` |
| R4 | Запропонувати та реалізувати модель мережі | **Feature Autoencoder (FAE)**: денойзинговий автоенкодер з «пляшковим горлом» над ознаками патчів (D6). Нормалізація ознак (fit на train) → GaussianNoise → Dense 256 → Dropout → Dense k (латент) → Dense 256 → Dense 512. Карта аномалій = MSE відновлення кожного патча → білінійний апсемплінг до 256×256 → гаусове згладжування σ = 4; оцінка зображення = максимум карти. Наскрізна `keras.Model(зображення → карта)` диференційовна | `tables/model_architecture.csv`, `figures/model_diagram.png` (схема «зображення → backbone → патчі → FAE → карта»), `models/<категорія>/fae_head.keras` |
| R5 | Експерименти з оптимізації регуляризації | Один-фактор-за-раз від базової конфігурації на `macaroni2` (D7): ширина латенту k ∈ {16, 64, 256}, Dropout ∈ {0, 0,2}, L2 ∈ {0, 1e-4}, шум входу σ ∈ {0, 0,2}; раннє зупинення для всіх. Критерій — Pixel AUROC на labelled val; додатково розрив втрат train / val_normal | `tables/study_regularization.csv`, `figures/study_regularization.png`, `metrics.studies.regularization.*` |
| R6 | Експерименти з вибору розміру патчів | Агрегація p ∈ {1, 3, 5} (поле зору патча: ≈ 8·p пікселів поверх receptive field) і розбиття на тайли 1×1 / 2×2 (D4), той самий backbone і FAE | `tables/study_patch_size.csv`, `figures/study_patch_size.png`, `metrics.studies.patch_size.*` |
| R7 | Порівняння різних моделей-екстракторів | FAE з однаковою конфігурацією на ResNet50 / MobileNetV2 / EfficientNetB0 (`macaroni2`); у таблиці — якість + GFLOPs + latency backbone | `tables/study_backbones.csv`, `figures/study_backbones.png`, `metrics.studies.backbones.*` |
| R8 | Навчити модель, відстежуючи втрати та метрики на train і val | Keras `fit` з `validation_data` (val_normal); колбек після кожної епохи рахує на labelled val Pixel AUROC, Image AUROC, Pixel AP і на train — p99 оцінок патчів (на train-нормі AUROC не визначений — лише один клас; це пояснюється у звіті). Раннє зупинення за `val_loss` з відновленням найкращих ваг. Історія — для всіх досліджень і фінальних моделей | `tables/training_history.csv` (усі запуски, ключ `run_id`), `figures/learning_curves.png` (фінальні моделі 3 категорій), `figures/learning_curves_study.png`, `metrics.training.<категорія>.*` |
| R9 | Метрики якості Pixel_AUROC, F1-Score | На test: Pixel AUROC (головна), Pixel F1 і Image F1 за порогами, обраними на labelled val (D3, D8); додатково Image AUROC, Pixel AP, «oracle» F1-max на test (лише для зіставлення з літературою, позначено окремо). 95 % бутстреп-ДІ по зображеннях | `tables/test_metrics.csv`, `figures/roc_pr_pixel.png`, `figures/anomaly_maps_test.png`, `metrics.test.<модель>.<категорія>.*` |
| R10 | Виміряти latency (мс/зобр) | Наскрізний конвеєр «зображення → карта» для FAE, PaDIM, PatchCore на кожному backbone: медіана й p90 після прогріву, batch = 1 і batch = 16, 1 потік і всі потоки (D10); окремо backbone і головка | `tables/latency.csv`, `metrics.profiling.*.latency_*` |
| R11 | Підрахувати FLOPS на GPU/CPU | FLOPs (= 2 × MAC) backbone + головки FAE — профайлером TF на замороженому графі; головки PaDIM / PatchCore — аналітично; плюс параметри і пам'ять (ваги, банк пам'яті, коваріації). GPU для TF недоступний (перевірено, D10) — вимірювання лише на CPU, чесно зазначено у звіті | `tables/flops.csv`, `figures/accuracy_vs_cost.png`, `metrics.profiling.*.gflops`, `metrics.profiling.device` |
| R12 | Порівняльний бенчмаркінг із SOTA (Anomalib: PaDIM, PatchCore) | Реімплементації PaDIM і PatchCore на **тих самих** TF-ознаках, розбитті й метриках (D9), явно названі «реімплементаціями»; плюс довідкові опубліковані числа (з посиланням на джерело, файлом-конфігом) для VisA. Парний бутстреп ДІ різниці Pixel AUROC FAE − SOTA | `tables/sota_benchmark.csv`, `tables/sota_bootstrap.csv`, `tables/published_reference.csv`, `figures/sota_benchmark.png`, `metrics.sota.*` |
| R13 | Підсумковий звіт про стан даних та виконані перетворення | Таблиця стану даних на кожному етапі (сирі → розбиття → resize → патчі/тайли → ознаки → нормалізація) з розмірностями і кількостями; перелік перетворень з параметрами і де вони «вчаться» (лише train) | `tables/data_overview.csv`, `tables/transformations.csv`, `tables/splits.csv`, `metrics.data.*` |

### 1.2. Проєктні вимоги (з `CLAUDE.md`, мети роботи і залежностей)

| ID | Вимога | Як буде виконано | Артефакт-доказ |
|---|---|---|---|
| P1 | Мета: «режим unsupervised / self-supervised» | FAE, PaDIM, PatchCore навчаються **лише на нормальних** train-зображеннях. Мітки аномалій використовуються тільки для вибору конфігурації й порогів (labelled val) і фінальної оцінки (test) — це явно названо у звіті (D3) | `tables/splits.csv` |
| P2 | Мета: «компроміс між точністю та обчислювальною складністю» | Графік Pixel AUROC проти GFLOPs і latency для всіх пар (головка × backbone); фронт Парето | `figures/accuracy_vs_cost.png`, `tables/accuracy_vs_cost.csv` |
| P3 | Відтворюваність | Seed з `LabContext.seed`: розбиття, вибір векторів на епоху, ініціалізація, випадкові канали PaDIM, проєкція і старт coreset PatchCore, бутстреп. Розбиття зберігається в `data/processed/visa/splits.csv` і перевикористовується ЛР5 | `metrics.config.*`, тест відтворюваності |
| P4 | `fit` лише на train, test не впливає на рішення | Нормалізація ознак, FAE, статистики PaDIM, банк PatchCore — лише train-нормальні; конфігурація і пороги — labelled val; test — один раз для фіксованого переліку моделей | тест «test не потрапляє у fit», огляд коду |
| P5 | Передача моделі в ЛР5 | Спільний пакет `mllabs.anomaly` (не `labs/lab04`): побудова наскрізної диференційовної моделі; збережені головки FAE, конфіг детектора (JSON), пороги, розбиття, прогнози test з позначками TP/FP/FN/TN і типом дефекту (D11) | `models/<категорія>/fae_head.keras`, `models/<категорія>/detector.json`, `tables/test_predictions.csv`, тест «наскрізна модель = кешований шлях» |
| P6 | `mllabs run 4` — до ~30–40 хв без навантаження (не рахуючи першого завантаження VisA) | Кеш зображень і ознак на диску, дослідження лише на `macaroni2`, обмежена кількість векторів на епоху, coreset 1 % (розд. 5.1) | `metrics.timing.*` |
| P7 | Жодних чисел «з голови» у звіті | Усе — з `metrics.json` і CSV; опубліковані числа — у `config` з посиланням на статтю і таблицю | `tables/published_reference.csv` |

## 2. Ключові рішення

### D1. Датасет — ПОТРЕБУЄ ПОГОДЖЕННЯ

| Варіант | Плюси | Мінуси |
|---|---|---|
| 1. **VisA** (Amazon, 2022) | офіційне пряме посилання S3 без облікових даних (перевірено); піксельні маски; 12 категорій, з них «multiple instances» і «complex structure» з неоднорідним тлом; дефекти дрібні → сильна піксельна незбалансованість; ліцензія CC BY 4.0 (за офіційним репозиторієм) | один архів 1,93 ГБ (потрібні лише 3 категорії); зображення великі (~1–1,5 Мпікс) → декодування на першому запуску; Anomalib у своєму коді вказує CC BY-NC-SA 4.0 — розбіжність, для навчальних цілей неважлива |
| 2. MVTec AD | класичний бенчмарк, найбільше опублікованих чисел | офіційне завантаження — через форму на сайті MVTec (не «без ручних кроків»), автопосилання Anomalib мертве (404); є лише неофіційні дзеркала на Hugging Face; CC BY-NC-SA; тло в більшості категорій однорідне — гірше відповідає «високій дисперсії тла» |
| 3. MVTec LOCO / BTAD | LOCO — логічні аномалії; BTAD — мала | LOCO — теж форма; логічні аномалії погано ловляться патч-методами; BTAD — 3 категорії, сумнівна стабільність посилань |

**Рекомендація: VisA.** Завантаження кодом (`VisaDatasetSource`): потоково качає архів у
`data/raw/visa/`, перевіряє SHA-256 `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362`,
розпаковує **лише** потрібні категорії і `split_csv/1cls.csv` (перевірено: файл є наприкінці архіву;
категорії в архіві йдуть за алфавітом). S3 підтримує HTTP Range (відповідь 206 перевірено) —
завантажувач має **докачувати** обірваний файл (`Range: bytes=<розмір .part>-`), бо при повільній
мережі завантаження триває десятки хвилин; файл пишеться як `.part` і перейменовується лише після
збігу SHA-256. Розпакування — `tarfile` з `filter="data"` (захист від шляхів поза каталогом).
Повторні запуски мережі не потребують. Якщо архів уже лежить у `data/raw/visa/` (з правильним
SHA-256), завантаження пропускається.

### D2. Категорії VisA — ПОТРЕБУЄ ПОГОДЖЕННЯ

Кількості за офіційним `1cls.csv`:

| Категорія | Тип (VisA) | train (норма) | test норма | test аномалії |
|---|---|---|---|---|
| macaroni2 | multiple instances, текстуроване неоднорідне тло | 900 | 100 | 100 |
| capsules | multiple instances, випадкові позиції | 542 | 60 | 100 |
| pcb4 | complex structure, 7 типів дефектів | 904 | 101 | 100 |

| Варіант | Плюси | Мінуси |
|---|---|---|
| 1. **3 категорії: macaroni2 + capsules + pcb4; дослідження гіперпараметрів лише на macaroni2** | три різні «види складності» (тло, кількість і положення об'єктів, структура); ≈ 2,9 тис. зображень — вкладається в бюджет; macaroni2 у літературі одна з найважчих (Image AU-ROC PaDiM 69,8 у Table 7 статті VisA) — є що досліджувати | конфігурація, обрана на macaroni2, може бути неоптимальною для інших двох (це й перевіряється — узагальнення) |
| 2. Усі 12 категорій | повний бенчмарк | ≈ 10,8 тис. зображень, ×4 час ознак і SOTA (> 1,5 год) |
| 3. Одна macaroni2 | найшвидше | висновки на одній категорії; 100 аномалій → дуже широкі ДІ |

**Рекомендація: варіант 1.** Вимірювання дисперсії тла (`background_variance.csv`) рахується для
всіх 12 категорій на 50 нормальних зображеннях кожної, щоб вибір був обґрунтований даними, а не
лише літературою. Технічно: архів читається потоково
**один раз** (`tarfile` в режимі ітерації): обрані категорії розпаковуються повністю, з решти — лише
перші 50 нормальних зображень (≈ 10–20 с додатково). Перелік категорій фіксується в `config.py`;
якщо виміряна дисперсія тла не підтвердить вибір, звіт має чесно це сказати (обґрунтування тоді —
складність структури й піксельна незбалансованість).

### D3. Розбиття та роль міток

Офіційне розбиття `1cls` не має валідаційної частини. Пропоноване (seed з контексту, стратифіковано):

| Частина | Звідки | Склад (macaroni2) | Для чого |
|---|---|---|---|
| `train` | 85 % офіційного train | ≈ 765 норм | fit нормалізації, FAE, PaDIM, PatchCore |
| `val_normal` | 15 % офіційного train | ≈ 135 норм | `val_loss` FAE, раннє зупинення, p99 оцінок норми |
| `val` (labelled) | 50 % офіційного test, страт. за міткою | ≈ 50 норм + 50 аномалій | вибір конфігурації в дослідженнях, пороги F1, метрики по епохах |
| `test` | решта 50 % офіційного test | ≈ 50 норм + 50 аномалій | лише фінальна оцінка |

| Варіант | Плюси | Мінуси |
|---|---|---|
| 1. **Labelled val + test (50/50 офіційного test)** | чесна оцінка; є дані для вибору гіперпараметрів і порогів; можна відстежувати Pixel AUROC по епохах | test менший (≈ 50 аномалій на категорію) → ширші ДІ; вибір за мітками = «напівнаглядова» селекція (явно назвати) |
| 2. Поріг за квантилем оцінок норми (val_normal), увесь офіційний test — test | повністю unsupervised | гіперпараметри обирати нема за чим: низький loss відновлення ≠ кращий детектор (широкий латент відновлює і дефекти); F1 сильно залежить від квантиля |
| 3. F1-max і вибір конфігурації одразу на test (як у багатьох статтях) | збіг із літературою | витік test у рішення — заборонено правилами проєкту |

**Рекомендація: варіант 1**, плюс колонка «oracle F1-max на test» (позначена як верхня межа, ні
для чого не використовується), щоб числа можна було зіставити з літературою. Нормальні зображення
test стратифікуються лише за міткою норма/аномалія: у `image_anno.csv` поле `label` для аномалій —
це **комбінації** типів дефектів (у capsules, напр., `bubble,discolor,scratch`), груп забагато для
стратифікації. Рядок типів зберігається в `splits.csv` для аналізу в ЛР5.

### D4. Патчі: вхідна роздільність, агрегація, тайли

- **Вхід:** усе зображення (без центр-кропу, щоб не втратити дефекти біля країв) → resize 256×256
  (білінійно; маска — nearest, бінаризація `> 0`). Пропорції VisA (не квадратні) спотворюються однаково
  для train і test — прийнятно, зазначити. Зменшені копії кешуються як `uint8` (≈ 570 МБ на 2,9 тис.).
- **Патч на рівні ознак:** вектор у позиції (i, j) сітки 32×32 (крок 8 пікс.). Локальна агрегація
  p×p (`AveragePooling2D(p, strides=1, padding="same")`) розширює «патч» без зміни сітки — це
  визначення розміру патча з PatchCore.
- **Явні тайли:** зображення ділиться на сітку g×g тайлів (`g = 2`), кожен тайл → 256×256 →
  ознаки; карти тайлів склеюються в карту 64×64 (ефективна роздільність 512). Це «розбиття
  зображень на патчі» у буквальному сенсі і типова практика для edge-інспекції високої роздільності.
- **Оцінювання** завжди на сітці 256×256 (карти тайлового варіанту зменшуються до 256), щоб метрики
  всіх варіантів рахувалися на однакових пікселях. Кількість масок, що стали порожніми після
  зменшення, записується (`metrics.data.<кат>.masks_empty_after_resize`).

Дослідження розміру патчів: p ∈ {1, 3, 5} при g = 1 + (g = 2, p = найкраще). Ознаки для g = 2
вилучаються лише для `macaroni2` і обраного backbone.

### D5. Екстрактори ознак

| Backbone | Шари (крок 8 / крок 16) | Канали | Нотатки |
|---|---|---|---|
| ResNet50 | `conv3_block4_out` / `conv4_block6_out` | 512 + 1024 | аналог layer2+layer3 PatchCore; препроцесинг `caffe` (BGR, віднімання середніх) |
| MobileNetV2 | `block_6_expand_relu` / `block_13_expand_relu` | 192 + 576 | edge-орієнтований (важливо для ЛР5); ваги ImageNet є лише для 96–224 — Keras підставить 224 з попередженням, мережа повністю згорткова, це коректно |
| EfficientNetB0 | `block4a_expand_activation` / `block6a_expand_activation` | 240 + 672 | препроцесинг вбудований (вхід 0–255) |

Злиття: шар кроку 16 → білінійний апсемплінг до 32×32 → конкатенація → **усереднення каналів до
D = 512** (фіксована лінійна матриця як `adaptive_avg_pool1d` у PatchCore, без навчання) → однакова
розмірність головок для всіх backbone, тож порівнюється саме екстрактор. Форми шарів перевірено
пробою. Альтернативи: ResNet101/ConvNeXt (важчі на CPU), WideResNet50 (стандарт PaDIM/PatchCore) —
відсутній у `keras.applications`; це відмінність від оригіналів, зазначити у звіті.

Вагові файли ImageNet Keras качає сам у `~/.keras/models` (≈ 100 МБ для ResNet50) — без облікових даних.

### D6. Пропонована модель — Feature Autoencoder (FAE)

| Варіант | Плюси | Мінуси |
|---|---|---|
| 1. **FAE: автоенкодер відновлення ознак патчів** (ідея DFR, Shi et al., 2021) | справжня мережа, що навчається (є втрати train/val, регуляризація має сенс); головка дешева (≈ 0,3 млн параметрів, константна пам'ять — на відміну від банку PatchCore) → цікавий компроміс для edge; reconstruction error — саме те, що потрібно для perturbation analysis у ЛР5; наскрізно диференційовна (IG, Grad-CAM) | очікувано поступиться PatchCore у якості — це нормальний, чесний результат (компроміс) |
| 2. Згортковий автоенкодер у просторі пікселів | класика | на VisA зі складним тлом слабкий; не використовує backbone (вимога 1); дорогий на CPU |
| 3. Student–Teacher (STFPM) | сильний, теж мережа | навчання студента — повна CNN на зображеннях щоепохи → години на CPU |
| 4. Гаусова модель ознак | простий | це й є PaDIM (SOTA-база), не «власна мережа» |

**Рекомендація: варіант 1.** Деталі:

- Головка працює на останній осі (`Dense` = згортка 1×1 = MLP на патч), тому **та сама модель**
  навчається на векторах `(N, 512)` і застосовується до карт `(B, 32, 32, 512)`.
- Шари: `Normalization` (адаптується на train-векторах) → `GaussianNoise(σ)` → `Dense(256, leaky_relu)`
  → `Dropout(r)` → `Dense(k)` → `Dense(256, leaky_relu)` → `Dense(512)`; `kernel_regularizer = L2(λ)`.
  Втрата — MSE у нормалізованому просторі.
- Карта = середній квадрат помилки по каналах → `tf.image.resize` до 256 (білінійно) → гаусове
  згладжування σ = 4 як depthwise-згортка в TF (не scipy — щоб градієнт проходив для ЛР5).
  Оцінка зображення = max карти. Та сама пост-обробка (TF) — для PaDIM і PatchCore.
- Навчання: Adam 1e-3, batch 512 векторів; на кожну епоху — випадкові 50 тис. векторів train
  (seed + номер епохи); `max_epochs = 30`; `EarlyStopping(val_loss, patience=5, restore_best_weights)`.
- Базова конфігурація досліджень: k = 64, Dropout 0, L2 0, σ_noise 0, p = 3, g = 1.

### D7. Протокол досліджень гіперпараметрів (лише `macaroni2`, вибір за labelled val)

Послідовно, кожне дослідження бере найкраще значення попереднього (жадібний координатний пошук —
найдешевший; це обмеження назвати у звіті):

1. **Екстрактори (R7):** FAE-база × {ResNet50, MobileNetV2, EfficientNetB0} — 3 запуски.
2. **Розмір патчів (R6):** p ∈ {1, 3, 5} (p = 3 уже є) + тайли g = 2 при найкращому p — 3 нові запуски.
3. **Регуляризація (R5):** один фактор за раз від бази: k ∈ {16, 256}; Dropout 0,2; L2 1e-4;
   σ_noise 0,2 (у нормалізованих одиницях) — 5 нових запусків.

Разом ≈ 11 навчань FAE. Критерій вибору — Pixel AUROC на val; при різниці < 0,002 — простіша/дешевша
конфігурація (менший backbone, менший p, g = 1). Кожен рядок таблиці дослідження: конфігурація,
Pixel AUROC / Pixel AP / Image AUROC на val, фінальні train loss і val_normal loss та їх розрив,
кількість епох, час навчання. Обрана конфігурація фіксується й застосовується до capsules і pcb4
без повторного підбору.

### D8. Метрики і пороги

- **Pixel AUROC** — головна (вимога). Через сильну піксельну незбалансованість (аномальних пікселів
  < 1 %) AUROC оптимістичний, тому поруч — **Pixel AP** (площа під PR-кривою), як PR-AUC у ЛР1–3.
- **F1** — двох рівнів: Image F1 (зображення аномальне, якщо max карти ≥ порога) і Pixel F1
  (сегментація). Пороги — максимум F1 на labelled val (`F1ThresholdSelector` з `mllabs.modeling`)
  → застосовуються до test без змін.
- **Image AUROC** — допоміжна.
- **ДІ:** бутстреп по зображеннях test (1000 повторів, ті самі індекси для всіх моделей → парні ДІ
  різниць). Для швидкості Pixel AUROC/AP у бутстрепі рахується з гістограм оцінок на зображення
  (спільні межі бінів, ~4096 квантильних бінів): повтор = зважена сума гістограм, без сортування
  мільйонів пікселів. Точні значення (без ДІ) — `sklearn.metrics`; тест перевіряє розбіжність < 1e-3.
  Та сама гістограмна функція — для метрик по епохах у колбеку.

### D9. SOTA-бенчмарк — ПОТРЕБУЄ ПОГОДЖЕННЯ

| Варіант | Плюси | Мінуси |
|---|---|---|
| (а) Встановити Anomalib (PyTorch) | «офіційні» реалізації, як у завданні; перевірено: встановлюється, імпортується, MPS доступний | +1,1 ГБ залежностей і другий фреймворк у проєкті; TF і PyTorch в одному процесі на macOS — ризик конфлікту OpenMP (потрібен окремий процес); інші backbone (WRN50/ResNet18 з timm) і інший конвеєр → порівнюються «конвеєри», а не головки; свій формат розбиття VisA (потрібен експорт карт для нашого test); навчання PatchCore WRN50 на CPU/MPS не перевірене за часом; новачку важче підтримувати |
| (б) **Реімплементація PaDIM і PatchCore на тих самих TF-ознаках** | чесне порівняння «за інших рівних» (той самий backbone, розбиття, пост-обробка, метрики) — відмінність лише в головці; прості алгоритми (≈ 100–150 рядків кожен); без нових залежностей; дає FLOPs/latency/пам'ять у тому самому середовищі | це не «бібліотечні» реалізації — у звіті називати «реімплементація PaDiM / PatchCore за статтями»; можливі дрібні відмінності (шари, D) |
| (в) Опубліковані числа | справжні SOTA-числа «з першоджерел» | інші протоколи (роздільність, кроп, розбиття, повний test) → лише довідка, не пряме порівняння; розбіжність між статтями велика (для macaroni2 PatchCore Image AUROC від ≈ 0,47 до ≈ 0,78 у різних роботах) |

**Рекомендація: (б) + (в)**, Anomalib не встановлювати. (б) — основне порівняння в `mllabs run 4`;
(в) — окрема довідкова таблиця з точним посиланням (стаття, таблиця, протокол), значення вносить
інженер у `config.py` з коментарем-джерелом. Перевірені джерела (2026-11-16):

- Zou et al., ECCV 2022, «SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and
  Segmentation» (arXiv 2207.14315, HTML v1), 1-class, ImageNet-supervised backbone, колонка
  «Supervised»: **Table 7** (PaDiM, класифікація, AU-PR / AU-ROC по підмножинах) і **Table 8** (PaDiM,
  сегментація, AU-PR / AU-ROC по підмножинах) — є рядки Macaroni2, Capsules, PCB4; **Table 6**
  (PatchCore, WRN50) — лише середні по VisA. Примітно: піксельний AU-PR PaDiM на macaroni2 там < 1 %
  при піксельному AU-ROC ≈ 0,96 — сильний аргумент, чому поруч з Pixel AUROC потрібен Pixel AP (D8).
  Середнє Image AU-ROC «Supervised» у Table 7 надруковано з опискою («7.80») — не використовувати.
- Для PatchCore по категоріях — таблиця VisA з DDAD (Mousakhan et al., arXiv 2305.15956, пари
  image / pixel AUROC для PaDiM і PatchCore); інженер має звірити значення з PDF (у пошуку вони
  траплялися, але точну таблицю я не відкривав), а для pcb4 рядка там може не бути.

Якщо автор захоче (а), це окреме розширення: група залежностей
`uv add --group sota anomalib torch torchvision`, окремий скрипт/процес, не частина бюджету `run 4`.

Деталі реімплементацій (на тих самих 32×32×512 ознаках з агрегацією p обраної конфігурації):

- **PaDIM** (Defard et al., 2020): випадкові d = 100 каналів (seed) → для кожної з 1024 позицій
  середнє і коваріація по train + ε·I (ε = 0,01) → обернена коваріація; оцінка — відстань
  Махаланобіса. Пам'ять ≈ 1024 × 100² × 4 Б ≈ 41 МБ.
- **PatchCore** (Roth et al., 2022): банк = усі вектори train (≈ 780 тис. для macaroni2) →
  жадібний k-center coreset 1 % (проєкція Джонсона–Лінденштраусса до 128 вимірів, seed); оцінка
  патча — відстань до найближчого сусіда в банку (батчами через матричне множення в TF); оцінка
  зображення — max (перезважування за сусідами зі статті — опційно, вимкнено, зазначити).
  Підказка для швидкості: відстань рахувати як `‖x‖² − 2x·c + ‖c‖²` (матрично-векторний добуток).
- Обидві — у спільному `mllabs.anomaly` (ЛР5 може використати як еталон для порівняння пояснень).

### D10. Профілювання: пристрій, FLOPs, latency

- **Пристрій:** лише CPU. `tensorflow-metal` несумісний з TF 2.21 (імпорт TF падає — перевірено),
  відкат TF до старої версії заради цього не виправданий. У звіті: «GPU для TensorFlow на цьому
  Mac недоступний; FLOPs не залежать від пристрою, latency виміряно на CPU». `metrics.profiling.device`
  містить модель CPU, кількість потоків, версії TF/Keras і load average під час вимірювання
  (`os.getloadavg()`) — щоб зафіксувати, якщо машина була завантажена.
- **FLOPs:** профайлер TF на замороженому графі `tf.function` моделі «зображення → ознаки (+ FAE)» з
  `batch=1` (FLOPs = 2 × MAC); перевірка: FLOPs головки FAE = Σ 2·in·out·1024 (тест). PaDIM:
  на позицію 2d² + 2d; PatchCore: на позицію 2·|M|·D (+ проєкція відсутня на інференсі). Пост-обробка
  (resize + blur) — теж профайлером, окремим рядком.
- **Latency:** `time.perf_counter`, 5 прогрівальних + 30 вимірювань, медіана і p90; batch 1
  (сценарій реального часу) і batch 16 (пропускна здатність, мс/зобр = час / 16); потоки:
  1 (наближення до слабкого edge-пристрою) і всі ядра. Декодування/resize зображення — окремий рядок.

### D11. Передача в ЛР5

Що потрібно ЛР5 і як це забезпечено:

- **Градієнти:** `mllabs.anomaly.detector.build_detector(config, head) -> keras.Model`, вхід —
  зображення 256×256×3 у діапазоні 0–255, виходи — карта 256×256 і оцінка зображення; препроцесинг
  backbone, злиття шарів, агрегація p, FAE, resize і blur — усе в TF. Grad-CAM потребує доступу до
  проміжного шару backbone — `build_detector` повертає також ім'я шарів-джерел ознак у конфігу.
- **Reconstruction error:** окремий вихід/метод `patch_errors` (сітка 32×32 до апсемплінгу).
- **Ваги:** `artifacts/models/<категорія>/fae_head.keras` (лише головка, кілька МБ; backbone
  відновлюється з `keras.applications` за назвою), `detector.json` (backbone, шари, D, p, g, σ blur,
  пороги image/pixel, seed, версії). Каталог `models/` у `.gitignore`, тож ЛР5 відтворює його
  запуском ЛР4 (документувати).
- **Розбиття:** `data/processed/visa/splits.csv` (категорія, шлях, маска, мітка, тип дефекту, частина).
- **FP/FN:** `tables/test_predictions.csv` — оцінка, поріг, рішення, TP/FP/FN/TN, тип дефекту, площа
  дефекту, σ яскравості тла зображення (для аналізу «неоднорідного тла» в ЛР5).
- Тест: на 2 зображеннях вихід наскрізної моделі збігається з «кешованим» шляхом (ознаки з кешу →
  головка → пост-обробка) з точністю ~1e-3 (кеш float16).

### D12. Кеш ознак

`data/processed/visa/features/<backbone>_g<g>/<категорія>.npy` (float16, `np.lib.format.open_memmap`)
+ JSON-маніфест (backbone, шари, D, розмір входу, версії TF/Keras, хеш списку файлів). Кеш
перевикористовується, якщо маніфест збігається, інакше перераховується. Ознаки зберігаються **до**
агрегації p (p застосовується на льоту — дешево). Обсяг: 1 МБ/зображення → ≈ 2,9 ГБ (обраний backbone,
3 категорії) + ≈ 2,2 ГБ (2 інші backbone, macaroni2) + ≈ 4,4 ГБ (g = 2, macaroni2) ≈ 9,5 ГБ на диску
(вільно 330 ГБ). Читання — через memmap, у пам'ять береться одна категорія. Пам'ять PatchCore:
повний банк macaroni2 ≈ 780 тис. × 512 × 4 Б ≈ 1,6 ГБ (float32) + проєкція 0,4 ГБ — прийнятно;
після coreset банк ≈ 16 МБ.

## 3. Дизайн коду

### 3.1. Спільний пакет `src/mllabs/anomaly/` (потрібен ЛР4 і ЛР5)

| Файл | Клас / функція | Відповідальність |
|---|---|---|
| `visa.py` | `VisaDatasetSource`, `VisaSample` (frozen dataclass: категорія, шлях, маска, мітка, тип дефекту), `DatasetDownloadError` (з `mllabs.data`) | завантаження з перевіркою SHA-256, вибіркове безпечне розпакування, читання `1cls.csv` і `image_anno.csv` |
| `splits.py` | `AnomalySplitter`, `AnomalySplit` | розбиття D3 (seed), збереження/читання `splits.csv` |
| `images.py` | `ImageStore` | resize до 256, маски, тайли g×g, кеш `uint8` у `data/processed/visa/images/` |
| `backbones.py` | `BackboneKind` (StrEnum), `BackboneSpec` (frozen: шари, препроцесинг), `FeatureExtractorFactory` | Keras-модель «0–255 зображення → 32×32×D» з `ChannelPooling` (власний серіалізовний шар) |
| `features.py` | `FeatureCache`, `PatchAggregator` | кеш D12; агрегація p×p |
| `postprocess.py` | `AnomalyMapPostprocessor` | resize до 256 + гаусове згладжування (TF depthwise conv) + оцінка зображення |
| `fae.py` | `FaeSettings` (frozen), `FeatureAutoencoderBuilder`, `FaeTrainer`, `EpochMetricsCallback` | модель D6, навчання з історією, метрики по епохах |
| `padim.py` | `Padim` | реімплементація (fit / score_maps) |
| `patchcore.py` | `PatchCore`, `GreedyCoresetSampler` | реімплементація (fit / score_maps) |
| `detector.py` | `PatchScorer` (Protocol: `fit(train_maps)`, `score(maps) -> patch_maps`), `DetectorConfig` (frozen, JSON), `build_detector(...)` | спільний інтерфейс трьох головок; наскрізна диференційовна модель для ЛР5 |
| `metrics.py` | `SegmentationMetrics` (frozen), `AnomalyMetricsCalculator`, `HistogramAuc`, `ImageBootstrap` | Pixel/Image AUROC, Pixel AP, F1 image/pixel за порогом; гістограмний бутстреп по зображеннях (повертає `BootstrapInterval` з `mllabs.modeling`) |
| `profiling.py` | `FlopsCounter`, `LatencyBenchmark`, `LatencyResult` | FLOPs (профайлер TF), latency (D10) |

Перевикористання: `ProjectPaths`, `ArtifactStore`, `LabContext.seed`; `F1ThresholdSelector`,
`BootstrapInterval` з `mllabs.modeling`; `DatasetDownloadError` з `mllabs.data`; стиль графіків
`mllabs.plotting` (`styled`, `FULL_WIDTH`/`HALF_WIDTH`, палітра, ч/б-маркери). `MetricsCalculator`
і `PairedBootstrap` не підходять напряму (рядки-пікселі мільйонами, групування по зображеннях) —
тому окремий `ImageBootstrap`, але з тим самим `BootstrapInterval`.

### 3.2. Пакет `src/mllabs/labs/lab04/`

| Файл | Вміст |
|---|---|
| `config.py` | `Lab4Config` (розд. 3.4) і `PublishedResult` (довідкові числа з джерелами) |
| `lab.py` | `Lab4` — лише оркестрація кроків |
| `data_overview.py` | `DataOverview`: таблиці стану даних, дисперсія тла, площа дефектів, перетворення |
| `studies.py` | `StudyRunner`, `StudyResult`: три дослідження D7, вибір конфігурації |
| `final_training.py` | `FinalTrainer`: навчання обраної FAE на 3 категоріях, збереження головок і `detector.json` |
| `evaluation.py` | `TestEvaluator`: пороги з val, метрики test, бутстреп, `test_predictions.csv` |
| `benchmark.py` | `SotaBenchmark`: PaDIM/PatchCore, парні ДІ, таблиця з опублікованими числами |
| `profiling_study.py` | `ProfilingStudy`: FLOPs і latency для всіх (головка × backbone), точність-проти-вартості |
| `plots.py` | усі графіки (`@styled`) |
| `summary.py` | зведені `metrics.json` і таблиці |
| `timing.py` | таймер кроків (за зразком `lab03/timing.py`; якщо однаковий — винести в `mllabs.core`) |

### 3.3. Кроки `Step` у `lab.py`

1. `1. Дані VisA: завантаження, розбиття, стан даних` — R1, R13 (кеш зображень 256 і тайлів).
2. `2. Патчі та ознаки backbone (кеш)` — R2, R3: обраний перелік (категорія × backbone × g).
3. `3. Дослідження екстракторів` — R7.
4. `4. Дослідження розміру патчів` — R6.
5. `5. Дослідження регуляризації` — R5.
6. `6. Навчання фінальної моделі (3 категорії)` — R4, R8; збереження для ЛР5.
7. `7. SOTA: PaDIM і PatchCore (реімплементації)` — R12 (fit на train, карти val/test).
8. `8. Оцінка на test` — R9, R12 (пороги з val; бутстреп; FP/FN).
9. `9. Профілювання FLOPs і latency` — R10, R11, P2.
10. `10. Зведення, графіки` — R13, P2.

Порядок рахування ознак: крок 2 гарантує кеш лише для `macaroni2` × 3 backbone (g = 1) — цього
досить для кроку 3. Ознаки g = 2 (крок 4) і ознаки capsules / pcb4 для **обраного** backbone (кроки
6–8; backbone відомий лише після кроку 3) рахуються ліниво через `FeatureCache.get_or_compute` у
тому кроці, де вперше потрібні. Ознаки capsules / pcb4 для необраних backbone не потрібні взагалі.

### 3.4. Параметри `config.py` (`Lab4Config`)

- Дані: `visa_url`, `visa_sha256`, `categories =
  ("macaroni2", "capsules", "pcb4")`, `study_category = "macaroni2"`, `train_fraction = 0.85`,
  `val_fraction_of_test = 0.5`, `background_sample_per_category = 50`.
- Зображення / патчі: `image_size = 256`, `feature_grid = 32`, `patch_sizes = (1, 3, 5)`,
  `default_patch_size = 3`, `tile_grids = (1, 2)`, `eval_size = 256`, `blur_sigma = 4.0`.
- Backbone: `backbones = (RESNET50, MOBILENET_V2, EFFICIENTNET_B0)`, `default_backbone = RESNET50`,
  `feature_dim = 512`, `extraction_batch_size = 16`.
- FAE (база): `hidden = 256`, `latent = 64`, `dropout = 0.0`, `l2 = 0.0`, `input_noise = 0.0`,
  `learning_rate = 1e-3`, `batch_size = 512`, `vectors_per_epoch = 50_000`, `max_epochs = 30`,
  `patience = 5`.
- Сітка регуляризації: `latent_grid = (16, 256)`, `dropout_grid = (0.2,)`, `l2_grid = (1e-4,)`,
  `noise_grid = (0.2,)`; `selection_tolerance = 0.002`.
- SOTA: `padim_channels = 100`, `padim_epsilon = 0.01`, `patchcore_coreset_ratio = 0.01`,
  `patchcore_projection_dim = 128`, `knn_batch = 4096`; `published_results` (кортеж `PublishedResult`
  зі статтею, таблицею, протоколом).
- Оцінка: `bootstrap_repeats = 1000`, `confidence_level = 0.95`, `histogram_bins = 4096`.
- Профілювання: `latency_warmup = 5`, `latency_runs = 30`, `latency_batch_sizes = (1, 16)`,
  `latency_threads = (1, None)` (None — усі ядра).
- Потоки TF: `intra_op_threads = None` (усі) — для навчання; встановлювати до першого використання TF.

### 3.5. Тести (`tests/anomaly/`, `tests/labs/test_lab04_*.py`) — на синтетичних даних, без мережі

- `VisaDatasetSource`: розбір `1cls.csv`/`image_anno.csv` на фікстурі; відмова при поганому SHA;
  безпечне розпакування (член з `../` відкидається).
- `AnomalySplitter`: стратифікація, детермінізм за seed, train/val/test не перетинаються,
  у train/val_normal немає аномалій.
- `ChannelPooling`: збіг з еталонною матрицею усереднення; `PatchAggregator` p = 1 — тотожність.
- `FlopsCounter`: FLOPs однієї `Dense` = 2·in·out (на позицію).
- `HistogramAuc` ≈ `roc_auc_score` / `average_precision_score` (< 1e-3); бутстреп відтворюваний.
- `Padim` / `PatchCore`: на синтетиці (нормальні вектори з N(0, I), аномальні зсунуті) аномальні
  оцінки більші; coreset повертає унікальні індекси потрібної кількості.
- FAE: модель, навчена на векторах, застосовується до карт тієї ж форми; seed → однакові ваги.
- «Наскрізна модель = кешований шлях» (D11) на малому backbone (`weights=None`, вхід 64×64) і
  `tf.GradientTape` дає ненульовий градієнт оцінки за входом (гарантія для ЛР5).
- `Lab4` на мікро-фікстурі (синтетичні зображення, малий backbone через конфіг) проходить усі кроки.

## 4. Артефакти (`labs/lab04/artifacts/`)

### 4.1. Графіки (`figures/`)

| Файл | Що показує |
|---|---|
| `samples_grid.png` | по 2 нормальних і 2 аномальних зображення кожної категорії з контуром маски |
| `background_variance.png` | дисперсія тла по 12 категоріях VisA, обрані — виділено |
| `defect_area_hist.png` | розподіл частки площі дефекту (лог-шкала) по категоріях |
| `patch_grid.png` | сітка 32×32 патчів, вікно p×p і тайли 2×2 на прикладі зображення |
| `model_diagram.png` | схема конвеєра з розмірностями |
| `study_backbones.png` | Pixel AUROC (val) і latency backbone для 3 екстракторів |
| `study_patch_size.png` | Pixel AUROC / AP від p, окремо g = 2 |
| `study_regularization.png` | Pixel AUROC (val) і розрив train−val_normal loss по варіантах |
| `learning_curves_study.png` | train/val_normal loss і val Pixel AUROC по епохах для досліджень |
| `learning_curves.png` | те саме для фінальних моделей 3 категорій |
| `roc_pr_pixel.png` | піксельні ROC і PR-криві FAE/PaDIM/PatchCore на test |
| `anomaly_maps_test.png` | зображення, маска, карти трьох методів (по 2 приклади на категорію, включно з FP і FN) |
| `sota_benchmark.png` | Pixel AUROC / Image AUROC з 95 % ДІ, три методи × три категорії |
| `accuracy_vs_cost.png` | Pixel AUROC проти GFLOPs і latency (batch 1), фронт Парето |

### 4.2. Таблиці (`tables/`)

`data_overview.csv`, `splits.csv`, `background_variance.csv`, `defect_area.csv`, `transformations.csv`,
`backbones.csv`, `model_architecture.csv`, `study_backbones.csv`, `study_patch_size.csv`,
`study_regularization.csv`, `training_history.csv`, `test_metrics.csv` (метод × категорія: Pixel
AUROC, Pixel AP, Pixel F1, Image AUROC, Image F1, пороги, oracle F1-max, ДІ), `test_predictions.csv`,
`sota_benchmark.csv`, `sota_bootstrap.csv`, `published_reference.csv`, `flops.csv`, `latency.csv`,
`accuracy_vs_cost.csv`.

### 4.3. Ключі `metrics.json`

- `data.<кат>.{n_train, n_val_normal, n_val_labelled_normal, n_val_labelled_anomalous,
  n_test_normal, n_test_anomalous, anomalous_pixel_share, defect_area_median, background_std, masks_empty_after_resize}`,
  `data.source.{url, sha256, license}`.
- `features.<backbone>.{layers, channels_raw, feature_dim, grid}`.
- `patches.{image_size, grid, stride_px, patch_sizes, tile_grids}`.
- `studies.backbones.<backbone>.{pixel_auroc, pixel_ap, image_auroc, epochs, train_seconds}`,
  `studies.patch_size.<p|g>.*`, `studies.regularization.<variant>.*`, `studies.selected.{backbone,
  patch_size, tile_grid, latent, dropout, l2, input_noise}`.
- `training.<кат>.{epochs, best_epoch, train_loss, val_loss, val_pixel_auroc, seconds}`.
- `test.<метод>.<кат>.{pixel_auroc, pixel_auroc_ci, pixel_ap, pixel_f1, image_auroc, image_f1,
  threshold_pixel, threshold_image, oracle_pixel_f1}`, `test.<метод>.mean.*`.
- `sota.delta.<кат>.{fae_minus_padim, fae_minus_patchcore}` (delta, low, high).
- `profiling.device.{cpu, threads, tf_version, keras_version, load_average}`,
  `profiling.<метод>.<backbone>.{gflops_backbone, gflops_head, params_head, memory_mb,
  latency_ms_b1_median, latency_ms_b1_p90, latency_ms_b16, latency_ms_b1_1thread}`.
- `config.*` (seed, усі параметри), `timing.<крок>` (секунди).

## 5. Ризики та обмеження

### 5.1. Час обчислень (CPU, Apple M1 Pro, без навантаження; оцінки за пробами під навантаженням)

| Етап | Оцінка |
|---|---|
| Перше завантаження VisA (1,93 ГБ) | **не входить у бюджет**: 5–40 хв залежно від мережі (у пробі ~0,5–1 МБ/с); далі — кеш |
| Розпакування 3 категорій + resize ≈ 2,9 тис. зображень (+ 50×12 для дисперсії тла) | 2–3 хв (перший запуск), далі секунди |
| Ознаки: ResNet50 для 2,9 тис. + MobileNetV2/EfficientNetB0 для 1,1 тис. + g = 2 (4,4 тис. тайлів) | 4–7 хв (перший запуск) |
| 11 навчань FAE (дослідження) + 2 фінальні (macaroni2 — перевикористати) | 12–18 хв (≤ 30 епох × ≈ 1,5 с навчання + ≈ 1 с метрик val) |
| PaDIM ×3 | ≈ 1 хв |
| PatchCore ×3 (coreset 1 %) | 4–6 хв |
| Оцінка, бутстреп (гістограмний), профілювання (≈ 9 пар × 2 batch × 2 режими потоків) | 3–5 хв |
| **Разом** | **≈ 27–40 хв** перший запуск після завантаження; повторний з кешем ознак ≈ 22–30 хв |

Під навантаженням (як зараз, load average > 100) усе у 3–4 рази довше. Регулятори, якщо бюджет
перевищено: `vectors_per_epoch` 50k → 30k, `max_epochs` 30 → 20, `patchcore_coreset_ratio` 0,01 → 0,005,
`latency_runs` 30 → 15, вимкнути `g = 2`.

### 5.2. Статистичні та методичні обмеження

- Test ≈ 50 аномальних зображень на категорію → ДІ Image AUROC широкі; Pixel AUROC стабільніший (багато
  пікселів), але пікселі одного зображення залежні — тому бутстреп по зображеннях, а не по пікселях.
- Конфігурацію обрано на одній категорії жадібним пошуком — можливий не глобальний оптимум і гірше
  перенесення на capsules / pcb4; це перевіряється і обговорюється.
- Вибір за labelled val — «напівнаглядова» селекція; моделі при цьому навчаються без міток.
- Числа реімплементацій PaDIM/PatchCore не дорівнюють опублікованим (інший backbone, шари, роздільність,
  розбиття) — порівнювати з (в) лише якісно.
- Pixel AUROC при ~1 % аномальних пікселів завищує враження — тому Pixel AP поруч.
- Latency на завантаженій машині ненадійна: фіксується load average; інженер має запускати фінальний
  `run 4` без паралельних важких процесів.
- Resize без кропу спотворює пропорції; дрібні дефекти після зменшення до 256 можуть зникнути з маски
  (лічильник `masks_empty_after_resize`).

### 5.3. Дані та залежності

- Нових Python-залежностей не потрібно (TF/Keras, numpy, scipy, sklearn, pillow уже є). Зображення
  читати через Pillow (вже встановлений транзитивно — рекомендую додати явною залежністю `uv add pillow`,
  оскільки код імпортує його напряму).
- Архів — недовірені дані: перевірка SHA-256, `tarfile.extractall(filter="data")`, лише потрібні члени.
- Ваги ImageNet Keras завантажуються з `storage.googleapis.com` при першому запуску (~100 МБ).
- Ліцензія VisA: CC BY 4.0 (офіційний репозиторій), Anomalib вказує CC BY-NC-SA 4.0 — у звіті
  навести обидва посилання; для навчальної роботи обидві дозволяють використання з атрибуцією.
- `tensorflow-metal` — не використовувати (несумісний з TF 2.21). Anomalib/PyTorch — не додавати
  (D9); якщо автор захоче — лише окремою групою залежностей і окремим процесом.

## Прийняті рішення (оркестратор, 2026-11-16)

Автор надав повну автономію; прийнято рекомендації плану:

- **D1** — VisA (офіційне посилання S3, перевірка SHA-256). Ліцензію (у джерелах CC BY 4.0 і
  CC BY-NC-SA 4.0) у звіті вказати з посиланням на офіційний репозиторій; для навчального
  використання підходять обидві.
- **D2** — macaroni2, capsules, pcb4; гіперпараметри — на macaroni2.
- **D9** — реімплементації PaDIM і PatchCore на тих самих TF-ознаках + опубліковані числа для
  довідки; Anomalib не встановлюємо. У звіті чітко назвати це реімплементаціями й пояснити,
  чому порівняння «головок» на спільному backbone чесніше за порівняння різних конвеєрів.
- **D3–D8, D10–D12** — як рекомендовано (Feature Autoencoder; ResNet50 / MobileNetV2 /
  EfficientNetB0; лише CPU; спільний пакет `src/mllabs/anomaly/` для ЛР5).
- Завантаження архіву VisA почато планувальником у scratchpad (`visa_dl/VisA_20220922.tar`);
  інженер переносить його в `data/raw/visa/` (докачати через Range за потреби, звірити SHA-256).
- Головний запуск — лише коли інші важкі обчислення не йдуть.
