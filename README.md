# Генерация синтетических данных на базе BlenderProc 2.7.1

Репозиторий содержит пайплайн генерации синтетических датасетов в Blender через **BlenderProc 2.7.1**.

Ключевые моменты:

- **Host (conda)**: запускает «ранеры» (`runers/*.py`), готовит конфиги и дергает `blenderproc run ...`.
- **Blender-Python (внутри BlenderProc)**: исполняет сценарии (`scenarios/*/main.py`).
- **blendforge** доступен **и на хосте**, и **в Blender-Python**.
- **BOP Toolkit** ставится из официального репо в Blender-Python, а ваши кастомные датасеты добавляются **патчем** (`bop_custom_patch.py`), который применяется автоматически при старте Blender-Python.

---

## 1) Требования

- **ОС:** Linux (проверено на Ubuntu 20.04 / 22.04)
- **Python:** 3.10 (рекомендуется `conda`)
- **GPU:** опционально (ускоряет рендеринг / OptiX)

---

## 2) Установка

### 2.1 Создать и активировать окружение conda

```bash
conda create -n gen python=3.10 -y
conda activate gen
```

### 2.2 Сделать скрипты исполняемыми

> Из корня репозитория:

```bash
chmod +x scripts/bproc scripts/blenderproc scripts/bop_autopatch_install.sh run.sh
```

### 2.3 Установить всё необходимое (host + Blender + Blender-Python)

```bash
scripts/bproc setup
```

По умолчанию Blender будет в:
`~/.cache/blenderproc-2.7.1/blender`

Если нужен кастомный путь:

```bash
BPROC_BLENDER_DIR=/path/to/blenderproc-cache scripts/bproc setup
```

## 3) Запуск

### Вариант A (рекомендуется): через `run.sh`

`run.sh` сам выставляет окружение (`eval "$(scripts/bproc env)"`) и запускает нужный раннер.

```bash
./run.sh
```

Дефолтные в `run.sh` пути можно переопределить флагами: 

```bash 
./run.sh --runner run_seg_bop.py --config configs/bop_seg/config.py
```

### Вариант B: вручную 

Перед запуском любых раннеров нужно один раз в текущей сессии выставить переменные окружения:

```bash
eval "$(scripts/bproc env)"
```

И после этого уже:

```bash
python runers/run_seg_bop.py --config_path configs/bop_seg/config.py
```
