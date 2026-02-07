import numpy as np 
import math


class TemperatureToRGBConverter:
    """
    Класс для конвертации температуры в цвет RGB, используя шкалу Кельвина или Цельсия.
    
    Этот класс позволяет преобразовать значение температуры в соответствующий цвет RGB,
    что может быть полезно для визуализации температурных карт или для задания цвета света
    в 3D-сценах в зависимости от температуры источника света.
    
    Attributes:
        temperature (float): Заданная температура для конвертации.
        scale (str): Используемая шкала измерения ('Kelvin' или 'Celsius').
        tmp_internal (float): Внутреннее значение температуры в Кельвинах для расчетов.
    
    Methods:
        __new__: Создает экземпляр класса и возвращает результат конвертации температуры в RGB.
        init: Инициализирует экземпляр заданными значениями температуры и шкалы.
        convert_to_rgb: Выполняет конвертацию температуры в цвет RGB.
        clamp: Ограничивает значение заданного аргумента указанными минимальным и максимальным значениями.
    """

    def __new__(cls, temperature: float, scale: str = 'Kelvin') -> np.ndarray:
        """
        Создает экземпляр и возвращает цвет RGB, соответствующий заданной температуре.

        Args:
            temperature (float): Температура.
            scale (str): Шкала измерения ('Kelvin' или 'Celsius').

        Returns:
            np.ndarray: Цвет в формате RGB, соответствующий заданной температуре.
        """
        instance = super(TemperatureToRGBConverter, cls).__new__(cls)
        instance.init(temperature, scale)
        return instance.convert_to_rgb()
    
    def init(self, temperature: float, scale: str = 'Kelvin') -> None:
        """
        Инициализирует экземпляр значениями температуры и шкалы измерения.
        
        Args:
            temperature (float): Заданная температура.
            scale (str): Шкала измерения ('Kelvin' или 'Celsius').
        
        Raises:
            AssertionError: Если указана неизвестная шкала измерения или температура вне допустимого диапазона.
        """
        self.scale = scale 
        assert scale in ['Kelvin', 'Celsius'], "Unknown scale"
        
        if self.scale == 'Kelvin':
            self.temperature = temperature
        else:
            self.temperature = temperature + 273.15
        
        assert 1000<= self.temperature <= 40000, "Temperature out of range. Range: 1000-40000 K"  

        self.tmp_internal = self.temperature / 100.0

    def convert_to_rgb(self) -> np.ndarray:
        """
        Производит конвертацию заданной температуры в цвет RGB.
        
        Returns:
            np.ndarray: Цвет в формате RGB, соответствующий заданной температуре.
        """
        # Red
        if self.tmp_internal <= 66:
            red = 255
        else:
            tmp_red = 329.698727446 * math.pow(self.tmp_internal - 60, -0.1332047592)
            red = self.clamp(tmp_red, 0, 255)

        # Green
        if self.tmp_internal <= 66:
            tmp_green = 99.4708025861 * math.log(self.tmp_internal) - 161.1195681661
            green = self.clamp(tmp_green, 0, 255)
        else:
            tmp_green = 288.1221695283 * math.pow(self.tmp_internal - 60, -0.0755148492)
            green = self.clamp(tmp_green, 0, 255)

        # Blue
        if self.tmp_internal >= 66:
            blue = 255
        elif self.tmp_internal <= 19:
            blue = 0
        else:
            tmp_blue = 138.5177312231 * math.log(self.tmp_internal - 10) - 305.0447927307
            blue = self.clamp(tmp_blue, 0, 255)

        return np.array([red / 255.0, green / 255.0, blue / 255.0])

    def clamp(self, x: float, min_val: float, max_val: float) -> float:
        """
        Ограничивает значение аргумента x заданными минимальным и максимальным значениями.
        
        Args:
            x (float): Значение для ограничения.
            min_val (float): Минимально допустимое значение.
            max_val (float): Максимально допустимое значение.
        
        Returns:
            float: Ограниченное значение x.
        """
        return max(min_val, min(x, max_val))