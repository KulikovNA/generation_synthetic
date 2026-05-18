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

Генерация RGB-D сегментации фрагментов (`seg_with_depth`):

```bash
./run.sh --runner run_seg_with_depth.py --config_path configs/seg_with_depth/config.py
```

Генерация RGB-D сегментации с активным стерео и несколькими depth-ветками:

```bash
./run.sh --runner run_seg_with_depth_stereo_multidepth.py --config_path configs/seg_with_depth/config_stereo_multidepth.py
```

Отладочный запуск stereo/multidepth с сохранением дополнительных промежуточных карт:

```bash
./run.sh --runner run_seg_with_depth_debug_stereo_multidepth.py --config_path configs/seg_with_depth/config_debug_stereo_multidepth.py
```

## 5) Сценарии генерации

Сценарии лежат в `scenarios/`. Каждый сценарий исполняется внутри BlenderProc, а соответствующий раннер из `runers/` готовит конфиг, временные директории, GPU-окружение и запускает `blenderproc run`.

Общая логика почти для всех сценариев одинаковая: берутся подготовленные BOP-модели из `prepared/<dataset>`, сцена собирается в простой комнате с рандомизированными CC0-текстурами, освещением, материалами объектов, позами камеры и физическим падением объектов. Параметр `probability_drop` переключает два режима размещения: "куча" после падения в небольшой области или более рассеянная раскладка. Это нужно, чтобы данные покрывали как плотные перекрытия объектов, так и более читаемые сцены.

### `scenarios/bop`

Базовая генерация BOP-датасета для задачи 6D pose estimation целых объектов. Сценарий формирует RGB, depth и BOP-разметку поз объектов относительно камеры.

- `main.py` - основной вариант для BOP: загружает целые объекты, рандомизирует PBR-материалы и свет, раскладывает объекты через физику, сэмплирует ракурсы вокруг сцены и пишет BOP-структуру через `bproc.writer.write_bop`.
- `main_stereo_multidepth.py` - BOP-вариант с активным стерео под профиль RealSense. Помимо GT depth формирует глубину из stereo matching для двух веток: `effective` projector и `random` projector pattern. Может сохранять IR-пары, а результат пишет в BOP-подобный multidepth-формат.
- `main_2.py` и `main_.py` - более ранние/экспериментальные варианты базовой BOP-генерации с другой логикой выбора объектов, камеры и материалов.

Этот набор полезен для обучения и проверки моделей оценки 6D-позы, которым нужны BOP-совместимые `scene_gt`, `scene_camera`, `rgb`, `depth`, маски и метаданные камеры.

### `scenarios/bop_seg`

Генерация данных для 2D instance segmentation и object detection по целым BOP-объектам.

- `main.py` пишет COCO-разметку: RGB-кадры, instance masks, категории объектов и bbox/segmentation.
- `main_yolo.py` пишет YOLO/Ultralytics-style сегментацию: `images`, `labels`, `classes.txt`, `data.yaml`, polygon masks. Именно этот вариант запускается текущим `runers/run_seg_bop.py`.

Особенности: объекты остаются целыми, материалы рандомизируются в сторону металлов и пластика, сцены специально включают перекрытия, падение в кучу и разные ракурсы. Сценарий нужен, когда модель должна находить и сегментировать объекты на RGB без обязательной 6D-разметки.

### `scenarios/deformed_bop_seg`

Сценарии для сегментации и дополнительных форматов по фрагментированным объектам. Исходные BOP-модели разрезаются через Blender Cell Fracture, после чего уже фрагменты раскладываются физикой и рендерятся.

- `main.py` пишет COCO instance segmentation по фрагментам.
- `grasp_net.py` пишет GraspNet-style структуру `scenes/scene_xxxx/...` с RGB, depth, label-картами, `camK.npy`, `camera_poses.npy` и списком object id. Текущий `runers/run_seg_bop_deformabel.py` запускает именно этот файл.
- `debug_scene.py` - отладочный сценарий для быстрой проверки фрактурирования, материалов, камеры и сегментации на одной сцене.

Смысл этого семейства - получить данные, где модель видит не целые объекты, а деформированные/разбитые фрагменты. Это ближе к задачам segment anything для деталей, grasp detection, анализа навалов и подготовки данных под пайплайны, где важны отдельные части объекта.

### `scenarios/fracture_6dpe`

Генерация BOP-подобного датасета для 6D pose estimation фрагментов и задачи "3D-пазла": восстановить положение фрагментов в сцене и в каноническом пространстве исходного объекта.

Сценарий фрактурирует объекты до размещения, экспортирует меши фрагментов и пишет расширенную структуру:

- `rgb/`, `depth/`, `scene_camera.json`, `scene_gt.json` - стандартная BOP-часть;
- `mask/`, `mask_visib/`, `scene_gt_info.json`, `scene_gt_coco.json` - маски, bbox и COCO-представление;
- `fragments/` - геометрия фрагментов;
- `fragments_gt.json` - соответствие фрагмента исходному объекту, локальный индекс, центр масс `com_F`, преобразование `R_O_from_F`, `t_O_from_F`;
- `scenes.json` - связь между "пазлом", набором фрагментов и кадрами, где эти фрагменты видны.

Данные из этого сценария подходят для оценки 6D-позы фрагментов, shape completion, сборки исходного объекта из частей и проверки глобальной согласованности фрагментов.

### `scenarios/seg_with_depth`

Семейство сценариев для RGB-D сегментации фрагментов. Оно объединяет COCO-разметку с картами глубины и дополнительными полями фрактурирования.

- `main.py` формирует `coco_annotations.json`, RGB-изображения, packed instance masks, чистую глубину `depth_inpainted`, "сырую" глубину `depth_raw` и `mask_valid`. Идея - обучать модели сегментации/детекции фрагментов с использованием глубины, а также проверять устойчивость к пропускам и шумам depth-сенсора.
- `main_stereo_multidepth.py` моделирует активное стерео RealSense: рендерит color camera и IR stereo, прогоняет stereo matching, выравнивает глубину в color frame и пишет несколько глубин: GT `depth`, `depth_effective`, `depth_random`. Дополнительно пишет YOLO-разметку в `yolo_data`.
- `debug_stereo_multidepth.py` повторяет stereo multidepth-пайплайн, но сохраняет дополнительные отладочные артефакты: raw left/right изображения, rectified depth, raw/filtered disparity previews и версии depth в `u16` или preview-gray режиме.
- `README.md` внутри папки подробнее описывает поля RGB-D COCO-формата.

Главная задача этого семейства - подготовка данных для моделей, которые должны одновременно сегментировать фрагменты и работать с глубиной: RGB-D segmentation, depth-aware detection, depth completion, проверка stereo/matcher-конфигов.

### `scenarios/bop_lol`

Генерация пар для low-light image enhancement в стиле LOL: для одной и той же сцены сохраняется тёмный `input` и нормальный/яркий `target`.

- `main.py` делает два рендера одной сцены: target при высокой энергии света и input при низкой энергии света. Результат пишется в `output_root/<split>/input`, `target`, `pairs.txt`, `meta.jsonl`.
- `main_low.py` - альтернативный вариант, где low-light получается постобработкой sRGB-кадра через EV-затемнение в линейном пространстве.

Такие данные нужны для обучения моделей улучшения изображений при слабом освещении и для доменной устойчивости детекторов/сегментаторов к недоэкспонированным кадрам. В метаданных сохраняются параметры камеры, рендера и экспозиции/режима генерации пары.
