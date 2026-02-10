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

### 2.4 Настройка рабочего пространства

Основные директории с предварительными данными 

```bash 
mkdir prepared resource output 
```

1. В файле конфигурации есть парметр отвечающий за входные данные `input_dataset_path` Эти данные долны соответствовать определенной структуре директории. Аналогично и подготовка этих данных требует на вход специфическую структуру директории. Предварительный набор данных создается с использованием `./bop_toolkit` и помещается в директорию `./prepared`

2. Для задания текстуры окружения и текстуры объектам используются библиотека текстур `CC0 Public Domain Textures`. А сами текстуры располагаются по пути `./resources/`. В файле конфигурации есть словарь `cc_textures` в котором переменная `cc_textures_plate` отвечает за текстуру для поверхности, а `cc_textures_obj` - текстуру объекта.


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

## 4) Запуск других проектов 

Генерация в формате LOL : 

```bash 
./run.sh --runner run_bop_lol.py --config_path configs/bop_lol/config.py 
```