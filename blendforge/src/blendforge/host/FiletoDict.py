import os
import json
import yaml
import ast
import os.path as osp
import re
import sys
import datetime
 
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence, Tuple, Union, Dict, List

class InvalidModeError(Exception):
    """
    Исключение, возникающее при предоставлении недопустимого режима.

    Args:
        mode (str): Режим, вызвавший исключение.
        message (str): Объяснение ошибки. По умолчанию "Режим не является допустимым".
    """
    
    def __init__(self, mode: str, message: str = "Режим не является допустимым"):
        self.mode = mode
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        """Возвращает строковое представление ошибки, указывающее недопустимый режим и сообщение."""
        return f'{self.mode} -> {self.message}'

class ConfigDict(dict):
    """
    Подкласс словаря, позволяющий доступ к его ключам как к атрибутам.

    Этот класс пытается имитировать объект конфигурации, где ключи можно получать как атрибуты.
    Если значение ключа является словарем, оно рекурсивно превращается в ConfigDict.
    """
    
    def __getattr__(self, key: str) -> Any:
        """
        Позволяет получать доступ к ключам словаря как к атрибутам.
        
        Args:
            key (str): Имя атрибута для доступа.
        
        Returns:
            Значение, ассоциированное с 'key' в словаре. Если значение является словарем, оно оборачивается в ConfigDict.
        
        Raises:
            AttributeError: Если ключ не существует.
        """
        try:
            value = super().__getitem__(key)
        except KeyError:
            raise AttributeError(f"'ConfigDict' объект не имеет атрибута '{key}'")
        if isinstance(value, dict):
            return ConfigDict(value)
        return value

    def __setattr__(self, key: str, value: Union[dict, Any]) -> None:
        """
        Позволяет устанавливать ключи словаря как атрибуты.

        Args:
            key (str): Имя атрибута для установки.
            value: Значение для установки данному ключу. Может быть любого типа.
        """
        self[key] = value


class Config:
    """
    Класс Config предназначен для анализа и валидации параметров конфигурации для генерации датасета.
    """

    def __init__(self, config_path: str):
        """
        Инициализирует класс Config.

        Args:
            config_path (str): Путь к файлу конфигурации. Поддерживаемые форматы: .py, .json, .yaml/.yml.
        """
        # Преобразование пути к абсолютному
        self.config_path = self._get_absolute_path(config_path)
        
        # Преобразование в dict
        self._data: Dict[str, Any] = self._load_config(self.config_path)
        
        # проверка переменных 
        self._validate_and_set_defaults()

    def __getattr__(self, key: str) -> Any:
        """
        Возвращает значение атрибута по ключу из данных конфигурации.

        Args:
            key (str): Ключ атрибута.

        Returns:
            Значение атрибута.

        Raises:
            AttributeError: Если атрибут с указанным ключом отсутствует.
        """

        if key in self._data:
            return self._data[key]
        raise AttributeError(f"'Config' object has no attribute '{key}'")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Загружает данные конфигурации из файла.

        Args:
            config_path (str): Путь к файлу конфигурации.

        Returns:
            dict: Данные конфигурации.
        """
        # Используйте локальную переменную вместо self.config_path
        self._check_path_exist(config_path)

        _, ext = os.path.splitext(config_path)
        
        # Проверка на соответствие формату
        if ext not in ['.py', '.json', '.yaml', '.yml']:
            raise OSError('Only py/yml/yaml/json type are supported now!')
        
        if ext == ".py":            
            config_data = self._load_py(config_path)
        
        elif ext == ".json":
            config_data = self._load_json(config_path)
        
        elif ext in [".yaml", ".yml"]:
            config_data = self._load_yaml(config_path)

        _data = self._to_config_dict(config_data) 
        
        return _data
    
    def _validate_and_set_defaults(self) -> None:
        """
        Валидирует параметры конфигурации и устанавливает значения по умолчанию для некоторых параметров.
        """

        required_params = {
            'num_gpus': int,
            'parallel_process_on_one_gpu': int,
            'mode': str, 
            'cc_textures': dict
        }
        
        for param, p_type in required_params.items():
            if param not in self._data:
                raise ValueError(f"Parameter {param} is required.")
            if not isinstance(self._data[param], p_type):
                raise TypeError(f"Parameter {param} must be of type {p_type.__name__}.")
        
        if self._data['cc_textures']: 
            
            for key, values in self._data['cc_textures'].items():   
                if not isinstance(values, str):
                    raise TypeError(f"Parameter {key} must be of type str.")        
                self._data['cc_textures'][key] = self._get_absolute_path(values)
        
        # если выходной директории нет - создаем
        if self._data['bop_dataset_name']: 
            self._create_output_dir(name_dataset = self._data['bop_dataset_name'])
        else: self._create_output_dir()

        # Менеджер мода  
        self._manager_mode()    
        
        # default для save_config 
        if 'save_config' not in self._data:
            self._data['save_config'] = True
        else: 
            if not isinstance(self._data['save_config'], bool):
                raise TypeError("Parameter save_config must be of type bool.")
        
        # default для 
        if 'random_camera_rotation' not in self._data or self._data['random_camera_rotation'] is None:
            self._data['random_camera_rotation'] = False

    def _manager_mode(self) -> None:
        """
        Управляет режимом работы конфигурации, вызывая соответствующий метод на основе указанного режима.

        Raises:
            InvalidModeError: Если указанный режим не поддерживается.
        """

        mode_name = self._data['mode']
        method_name = f'_mode_is_{mode_name}'

        try:
            # Получаем метод и вызываем его
            mode_method = getattr(self, method_name)
            mode_method()
        except AttributeError:
            # Если метод не найден, вызываем исключение
            raise InvalidModeError(mode_name)
    
    def _mode_is_deformed_bop_seg(self) -> None: 
        print(self._data)
        if self._data['dataset_parent_path']:
            self._data['dataset_parent_path'] = self._get_absolute_path(self._data['dataset_parent_path'])
            #self._data['output_dir'] = self._get_absolute_path(self._data['output_dir'])
        else: raise ValueError("dataset_parent_path не может быть пустым")

    def _mode_is_fracture_6dpe(self) -> None: 
        print(self._data)
        if self._data['dataset_parent_path']:
            self._data['dataset_parent_path'] = self._get_absolute_path(self._data['dataset_parent_path'])
            #self._data['output_dir'] = self._get_absolute_path(self._data['output_dir'])
        else: raise ValueError("dataset_parent_path не может быть пустым")

    def _mode_is_bop_lol(self) -> None:
        """
        Режим генерации парного датасета LLIE (LOL-style) с BOP-объектами.
        Готовит поля: dataset_parent_path, bop_toolkit_path, split, ev_low_range,
                    runs, poses_cam, probability_drop, max_amount_of_samples,
                    output_dir_lol.
        """
        print(self._data)

        # --- обязательные пути ---
        for key in ['dataset_parent_path', 'bop_dataset_name']:
            if not self._data.get(key):
                raise ValueError(f"Parameter '{key}' is required for mode 'bop_lol'.")

        # абсолютные пути
        self._data['dataset_parent_path'] = self._get_absolute_path(self._data['dataset_parent_path'])
        # bop_dataset_name — имя датасета (без абсолютизации)

        # --- split ---
        split = self._data.get('split', 'train')
        if split not in ('train', 'val', 'test'):
            raise ValueError("Parameter 'split' must be one of ['train','val','test'].")
        self._data['split'] = split

        # --- число сцен и поз ---
        self._data['runs']      = int(self._data.get('runs', 1))
        self._data['poses_cam'] = int(self._data.get('poses_cam', 25))
        if self._data['runs'] < 1 or self._data['poses_cam'] < 1:
            raise ValueError("'runs' and 'poses_cam' must be >= 1.")

        # --- вероятность «кучи» ---
        p_drop = float(self._data.get('probability_drop', 0.3))
        if not (0.0 <= p_drop <= 1.0):
            raise ValueError("'probability_drop' must be in [0,1].")
        self._data['probability_drop'] = p_drop

        # --- samples per frame ---
        mas = self._data.get('max_amount_of_samples', None)
        if mas is not None and not isinstance(mas, int):
            raise TypeError("'max_amount_of_samples' must be int or None.")
        self._data['max_amount_of_samples'] = mas

        # --- корень LOL-ветки ---
        # если задан output_dir_lol — используем его,
        # иначе создаём <output_dir>/lol/<YYYY-MM-DD>
        out_lol = self._data.get('output_dir_lol', None)
        if out_lol is None:
            base_out = self._data['output_dir']   # создаётся раньше в _create_output_dir
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            out_lol = os.path.join(base_out, 'lol', today)
        self._data['output_dir_lol'] = str(Path(out_lol).resolve())


    def _mode_is_seg_with_depth(self)->None:
        print(self._data)
        if self._data['dataset_parent_path']:
            self._data['dataset_parent_path'] = self._get_absolute_path(self._data['dataset_parent_path'])
            #self._data['output_dir'] = self._get_absolute_path(self._data['output_dir'])
        else: raise ValueError("dataset_parent_path не может быть пустым")

    def _mode_is_seg_with_depth_stereo_multidepth(self) -> None:
        print(self._data)
        if self._data['dataset_parent_path']:
            self._data['dataset_parent_path'] = self._get_absolute_path(self._data['dataset_parent_path'])
        else:
            raise ValueError("dataset_parent_path не может быть пустым")

    def _mode_is_bop_seg(self) -> None:
        print(self._data)
        if self._data['dataset_parent_path']:
            self._data['dataset_parent_path'] = self._get_absolute_path(self._data['dataset_parent_path'])
            #self._data['output_dir'] = self._get_absolute_path(self._data['output_dir'])
        else: raise ValueError("dataset_parent_path не может быть пустым")

    def _mode_is_bop(self)-> None: 
        print(self._data)
        if self._data['bop_parent_path']:
            self._data['bop_parent_path'] = self._get_absolute_path(self._data['bop_parent_path'])
            #self._data['output_dir'] = self._get_absolute_path(self._data['output_dir'])
        else: raise ValueError("bop_parent_path не может быть пустым")

    def _mode_is_bop_stereo_multidepth(self) -> None:
        print(self._data)
        if self._data['dataset_parent_path']:
            self._data['dataset_parent_path'] = self._get_absolute_path(self._data['dataset_parent_path'])
        else:
            raise ValueError("dataset_parent_path не может быть пустым")

    def _mode_is_classic(self) -> None:
        """
        Настраивает параметры конфигурации для классического режима работы.
        """

        input_dataset = self._data['input_dataset']

        # Логика для input_dataset и связанных параметров
        input_params = ['input_dataset_path', 'cam_params', 'models_info_path']
        input_values = {param: input_dataset.get(param, None) for param in input_params}
        if input_values['input_dataset_path']:
            input_dataset_path = self._get_absolute_path(input_values['input_dataset_path'])
            
            # Создаем списки для models_info_path и model_dir
            models_info_list = [str(self._get_absolute_path(os.path.join(input_dataset_path, "models", "models_info.json")))]
            
            # Устанавливаем значения по умолчанию
            self._data['input_dataset'].update({
                'cam_params': str(self._get_absolute_path(os.path.join(input_dataset_path, "camera.json"))),
                'models_info_path': models_info_list
            })
            
            if input_values[input_params[2]] is not None:
                self._data['input_dataset'][input_params[2]] = str(self._get_absolute_path(input_values[input_params[2]]))
            for param in input_params[2:]:
                if input_values[param] is not None:
                    self._data['input_dataset'][param] = [str(self._get_absolute_path(path)) for path in input_values[param]] 
            

    def _mode_is_crate(self) -> None:
        """
        Настраивает параметры конфигурации для режима работы с крейтами.

        Raises:
            ValueError: Если не удается найти файлы по указанным путям или если требуемые параметры отсутствуют.
        """

        input_dataset = self._data['input_dataset']
        # Логика для input_dataset и связанных параметров
        input_params = ['input_dataset_path', 'cam_params', 'crate_info_path', 'models_info_path']
        input_values = {param: input_dataset.get(param, None) for param in input_params}
        
        if input_values['input_dataset_path']:
            #путь
            input_dataset_path = self._get_absolute_path(input_values['input_dataset_path'])

            # Устанавливаем пути до файлов
            default_cam_params_path = os.path.join(input_dataset_path, "camera.json")
            default_crate_info_path = os.path.join(input_dataset_path, "crates_info.json")
            
            try:
                # Пытаемся получить абсолютный путь к пользовательскому значению cam_params или к значению по умолчанию
                if input_values['cam_params']:
                    cam_params_path = self._get_absolute_path(input_values['cam_params'])
                else:
                    cam_params_path = self._get_absolute_path(default_cam_params_path)
            except FileNotFoundError:
                # Если файл не найден, выбрасываем исключение
                raise ValueError("Не удалось найти файл cam_params ни по пользовательскому пути, ни по пути по умолчанию")

            # Устанавливаем значение cam_params
            self._data['input_dataset']['cam_params'] = cam_params_path

            try:
                if input_values['crate_info_path']:
                    crate_info_path = self._get_absolute_path(input_values['crate_info_path'])
                else:
                    crate_info_path = self._get_absolute_path(default_crate_info_path)
            except FileNotFoundError:
                raise ValueError("Не удалось найти файл crates_info.json ни по пользовательскому пути, ни по пути по умолчанию")

            self._data['input_dataset']['crates_info_path'] = crate_info_path

            crate_info = self._load_json(crate_info_path)

             # загружаем перечень крейтов 
            if 'crate_list' not in self._data:
                raise ValueError(f"Parameter 'crate_list' is required.")
            else: 
                if self._data['crate_list'] is None: 
                    filtered_crate_info = crate_info
                else: 
                    crate_list = self._data['crate_list']
                    # фильтруем crate_info в соответствии с crate_list    
                    filtered_crate_info = {crate: crate_info[crate] for crate in crate_list if crate in crate_info}
                    
            try:
                if input_values['models_info_path']:
                    models_info_path = {crate_name: self._get_absolute_path(input_values['models_info_path'][crate_name]) 
                                        for crate_name in filtered_crate_info.keys() 
                                        if crate_name in input_values['models_info_path']}
                else:
                    models_info_path = {crate_name: self._get_absolute_path(os.path.join(input_dataset_path, crate_info['models_info_path']))
                                        for crate_name, crate_info in filtered_crate_info.items()}
            except FileNotFoundError as e:
                # В этом случае сообщение об ошибке уже включено в исключение
                raise ValueError(f"Ошибка при поиске файла моделей: {e}")
            
            self._data['input_dataset']['models_info_path'] = models_info_path
            
        else: 
            raise ValueError("input_dataset_path не может быть пустым")


    def _create_output_dir(self, name_dataset = None) -> None:
        """
        Создает выходную директорию для результатов, если она не указана.

        Raises:
            OSError: Если возникают проблемы при проверки пути.
        """
        
        script_path = os.path.dirname(sys.argv[0])
        if name_dataset is None: 
            name_dataset = os.path.basename(self._data['input_dataset']['input_dataset_path']) 

        if 'output_dir' not in self._data:
            self._data['output_dir'] = None
        
        if self._data['output_dir'] is None:
            full_script_path = self._get_absolute_path(script_path)
            # Получение текущей даты и времени
            current_date = datetime.datetime.now()
            # Форматирование даты и времени в строку (например, '2024-01-09_15-30-00')
            time_str = current_date.strftime("%Y-%m-%d")
            output_dir = os.path.join(full_script_path, "output", name_dataset, time_str)
        else:
            absolute_path = Path(self._data['output_dir']).resolve()
            
            # Проверка, соответствует ли абсолютный путь формату '.../name_dataset/YYYY-MM-DD'
            pattern = re.compile(r'.*/' + re.escape(name_dataset) + r'/\d{4}-\d{2}-\d{2}$')
            if pattern.match(str(absolute_path)):
                output_dir = str(absolute_path)
            else:
                # Если путь не соответствует ожидаемому формату, создаем новую директорию
                current_date = datetime.datetime.now()
                time_str = current_date.strftime("%Y-%m-%d")
                output_dir = os.path.join(absolute_path, name_dataset, time_str)


        self._data['output_dir'] = output_dir
         

    def _to_config_dict(self, d: Dict[str, Union[dict, Any]]) -> 'ConfigDict':
        """
        Рекурсивно преобразует обычные словари в ConfigDict.

        Args:
            d (Dict[str, Union[dict, Any]]): Исходный словарь для преобразования.

        Returns:
            ConfigDict: Преобразованный словарь в ConfigDict.
        """

        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = self._to_config_dict(value)
        return ConfigDict(d)
    
    def _get_absolute_path(self, path_to_file_or_dir: str) -> str:
        """
        Возвращает абсолютный путь к указанному файлу или директории.

        Args:
            path_to_file_or_dir (str): Относительный или абсолютный путь к файлу или директории.

        Возвращает:
            str: Абсолютный путь к файлу или директории.

        Raises:
            FileNotFoundError: Если путь не существует.
        """

        # абсолютный путь
        absolute_path = Path(path_to_file_or_dir).resolve()
        # проверяем его наличие
        self._check_path_exist(absolute_path)

        return str(absolute_path)


    def _check_path_exist(self, path: Union[str, Path], msg_tmpl: str = 'Путь "{}" не существует') -> None:
        """
        Проверяет существование указанного пути.

        Args:
            path (Union[str, Path]): Путь для проверки.
            msg_tmpl (str): Шаблон сообщения об ошибке, используемый при генерации исключения.

        Raises:
            FileNotFoundError: Если указанный путь не существует.
        """

        if not (osp.isfile(path) or osp.isdir(path)):
            raise FileNotFoundError(msg_tmpl.format(path))
    
    def _validate_py_syntax(self, filename: str):
        """
        Проверяет синтаксис файла конфигурации Python.

        Args:
            filename (str): Имя файла конфигурации Python.

        Raises:
            SyntaxError: Если в файле конфигурации обнаружены синтаксические ошибки.
        """

        with open(filename, encoding='utf-8') as f:
            content = f.read()
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config '
                              f'file {filename}: {e}')

    def _load_py(self, path: str) -> Dict[str, Any]:
        """
        Загружает и выполняет Python файл конфигурации.

        Args:
            path (str): Путь к файлу конфигурации Python.

        Returns:
            Dict[str, Any]: Словарь с данными конфигурации.

        Raises:
            SyntaxError: Если в файле конфигурации обнаружены синтаксические ошибки.
        """

        self._validate_py_syntax(path)

        with open(path) as f:
            config_str = f.read()
        
        config_data: Dict[str, Any] = {}
        exec(config_str, config_data)

        # Фильтрация атрибутов, которые не начинаются с '__'
        return {key: value for key, value in config_data.items() if not key.startswith('__')}

    def _load_json(self, path: str) -> Dict[str, Any]:
        """
        Загружает JSON файл конфигурации.

        Args:
            path (str): Путь к файлу конфигурации JSON.

        Returns:
            Dict[str, Any]: Словарь с данными конфигурации.

        Raises:
            json.JSONDecodeError: Если файл содержит ошибки формата JSON.
        """

        with open(path, "r") as f:
            return json.load(f)
    
    def _load_yaml(self, path: str) -> Dict[str, Any]:
        """
        Загружает YAML файл конфигурации.

        Args:
            path (str): Путь к файлу конфигурации YAML.

        Returns:
            Dict[str, Any]: Словарь с данными конфигурации.

        Raises:
            yaml.YAMLError: Если файл содержит ошибки формата YAML.
        """

        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _add_new_item(self, nested_key: Optional[str] = None, **params: Any) -> None:
        """
        Добавляет новые параметры к текущему конфигу.

        Args:
            nested_key (Optional[str]): Ключ для вложенного словаря, который нужно обновить. 
                                        Если None, обновляется корневой словарь.
            **params: Параметры в виде ключ-значение для добавления.

        Raises:
            TypeError: Если указанный nested_key существует и не является словарем.
        """
        
        if nested_key:
            if nested_key in self._data:
                # Если ключ уже существует
                if isinstance(self._data[nested_key], dict):
                    # Если это словарь, обновляем его содержимое
                    self._data[nested_key].update(params)
                else:
                    # Если это не словарь, обновляем значение напрямую
                    self._data[nested_key] = SimpleNamespace(**params) if params else None
            else:
                # Если ключа нет, создаем новый SimpleNamespace или словарь
                self._data[nested_key] = SimpleNamespace(**params) if params else None
        else:
            self._data.update(params)




"""cfg = Config("/home/kulikov/testblend/configs/config_p.py")
print(cfg.input_dataset.models_info_path)"""
