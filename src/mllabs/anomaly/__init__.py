"""Спільний пакет детекції аномалій на зображеннях (ЛР4 — навчання, ЛР5 — пояснення).

Модулі важкі (імпортують TensorFlow), тому тут нічого не реекспортується: імпортуйте
потрібне прямо з модуля, напр. ``from mllabs.anomaly.detector import load_bundle``.

Як ЛР5 відтворює детектор ЛР4 (каталог ``labs/lab04/artifacts/models/<категорія>``)::

    from mllabs.anomaly.detector import load_bundle

    config, detector = load_bundle(models_dir / "macaroni2", expose_backbone=True)
    outputs = detector(images)        # images: (B, 256, 256, 3), float32, 0–255
    outputs["anomaly_map"]            # (B, 256, 256) карта аномалій
    outputs["image_score"]            # (B,) оцінка зображення (> config.threshold_image → аномалія)
    outputs["patch_scores"]           # (B, 32, 32) помилка відновлення FAE кожного патча
    outputs["backbone_shallow"]       # активації шару config.layers[0] (крок 8) — для Grad-CAM
    outputs["backbone_deep"]          # активації шару config.layers[1] (крок 16)
    detector.get_layer("fae_head")    # головка FAE: ознаки (B, h, w, D) → помилка (B, h, w)
    detector.get_layer("backbone")    # обрізана мережа ImageNet

Grad-CAM: у ``tf.GradientTape`` викликати ``detector(images)`` і взяти
``tape.gradient(outputs["image_score"], outputs["backbone_deep"])``.

Стабільний API (на нього спирається ЛР5; не перейменовувати):

* імена шарів наскрізної моделі: ``backbone`` (обрізана мережа ImageNet), ``fae_head``
  (головка FAE) з вкладеними ``normalization`` і ``fae_core`` (автоенкодер), а також
  ``channel_pooling``, ``postprocess``;
* ``feature_extractor`` — ім'я моделі «зображення → ознаки» з
  ``FeatureExtractorFactory.build`` (у наскрізному детекторі її шари вбудовано напряму,
  без вкладеної моделі, щоб активації ``backbone`` були доступні для Grad-CAM);
* виходи детектора: ``anomaly_map``, ``image_score``, ``patch_scores``,
  ``backbone_shallow``, ``backbone_deep`` (два останні — з ``expose_backbone=True``);
* дані: ``mllabs.anomaly.workspace.VisaWorkspace`` + ``WorkspaceSettings`` (розбиття
  з ``data/processed/visa/splits.csv``, кеші зображень і ознак).
"""
