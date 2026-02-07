import bpy
import random
import numpy as np
from typing import Iterable, Optional, Tuple

import os
from typing import List, Optional, Dict, Tuple

from blenderproc.python.material import MaterialLoaderUtility
from blenderproc.python.types.MaterialUtility import Material
from blenderproc.python.utility.Utility import resolve_path
from blenderproc.python.loader.CCMaterialLoader import _CCMaterialLoader

try:
    import blenderproc as bproc
    BPROC = True
except Exception:
    BPROC = False

MATERIAL_STYLES = (
    "metal",
    "dirty_metal",
    "rusty_metal", # не оч 

    "steel",
    "stainless_steel", # не оч
    "brushed_steel",
    "galvanized_steel",
    "blackened_steel",

    "cast_iron",

    "plastic_new",
    "plastic_old",
)

# ---------------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------------------

def _new_mat(name: str):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in nt.nodes:
        nt.nodes.remove(n)
    # Базовые узлы
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (300, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat, nt, bsdf, out

def _texcoord_mapping(nt, loc=(-1200, 0)):
    tex = nt.nodes.new("ShaderNodeTexCoord")
    tex.location = (loc[0], loc[1])
    mapn = nt.nodes.new("ShaderNodeMapping")
    mapn.location = (loc[0] + 200, loc[1])
    nt.links.new(tex.outputs["Object"], mapn.inputs["Vector"])
    return tex, mapn

def _noise(nt, scale=5.0, detail=8.0, rough=0.5, loc=(-900, 0)):
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.location = loc
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = rough
    return n

def _musgrave(nt, scale=6.0, detail=8.0, dimension=1.0, loc=(-900, -200)):
    m = nt.nodes.new("ShaderNodeTexMusgrave")
    m.location = loc
    m.inputs["Scale"].default_value = scale
    m.inputs["Detail"].default_value = detail
    m.inputs["Dimension"].default_value = dimension
    return m

def _voronoi(nt, scale=10.0, loc=(-900, -400)):
    v = nt.nodes.new("ShaderNodeTexVoronoi")
    v.location = loc
    v.inputs["Scale"].default_value = scale
    return v

def _ramp(nt, loc=(-700, 0), p0=0.2, p1=0.8, c0=(0,0,0,1), c1=(1,1,1,1)):
    r = nt.nodes.new("ShaderNodeValToRGB")
    r.location = loc
    r.color_ramp.elements[0].position = p0
    r.color_ramp.elements[1].position = p1
    r.color_ramp.elements[0].color = c0
    r.color_ramp.elements[1].color = c1
    return r

def _bump(nt, strength=0.1, dist=1.0, loc=(-200, -200)):
    b = nt.nodes.new("ShaderNodeBump")
    b.location = loc
    b.inputs["Strength"].default_value = strength
    b.inputs["Distance"].default_value = dist
    return b

def _mix_shader(nt, loc=(300, -200)):
    m = nt.nodes.new("ShaderNodeMixShader")
    m.location = loc
    return m

def _ao(nt, loc=(-1200, -300)):
    ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
    ao.location = loc
    return ao

def _geometry(nt, loc=(-1200, -100)):
    g = nt.nodes.new("ShaderNodeNewGeometry")
    g.location = loc
    return g

def _hsv(nt, loc=(-500, 200), h=0.0, s=1.0, v=1.0):
    node = nt.nodes.new("ShaderNodeHueSaturation")
    node.location = loc
    node.inputs["Hue"].default_value = h
    node.inputs["Saturation"].default_value = s
    node.inputs["Value"].default_value = v
    return node

def _color(base_col=None):
    # Набор сдержанных базовых цветов
    if base_col is None:
        palette = [
            (0.32, 0.20, 0.05),    # тёплый коричневый
            (0.18, 0.18, 0.18),    # тёмно-серый
            (0.55, 0.02, 0.02),    # тёмно-красный (лак)
            (0.08, 0.22, 0.38),    # холодный синий
            (0.28, 0.28, 0.30)     # графит
        ]
        base_col = random.choice(palette)
    return (*base_col, 1.0)

# ---------------------- МАТЕРИАЛЫ: ФАБРИКИ --------------------------

def make_metal(name="Metal_Default"):
    mat, nt, bsdf, out = _new_mat(name)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Specular"].default_value = 0.5
    bsdf.inputs["Roughness"].default_value = np.clip(np.random.uniform(0.1, 0.35), 0.05, 0.6)
    # Цвет отражения металла (слегка тонированный)
    col = random.choice([(0.86,0.86,0.86), (0.95,0.80,0.6), (0.80,0.85,0.92)])
    bsdf.inputs["Base Color"].default_value = (*col, 1.0)
    # Микрошероховатость
    tex, mapn = _texcoord_mapping(nt, (-1200, 0))
    n = _noise(nt, scale=np.random.uniform(8,14), detail=np.random.uniform(4,8), rough=np.random.uniform(0.3,0.6), loc=(-950, 0))
    nt.links.new(mapn.outputs["Vector"], n.inputs["Vector"])
    ramp = _ramp(nt, (-750, 0), p0=0.3, p1=0.7)
    nt.links.new(n.outputs["Fac"], ramp.inputs["Fac"])
    bump = _bump(nt, strength=np.random.uniform(0.03, 0.12), dist=1.0, loc=(-450, -100))
    nt.links.new(ramp.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return bproc.types.Material(mat) if BPROC else mat

def make_dirty_metal(name="Metal_Dirty"):
    mat, nt, bsdf, out = _new_mat(name)
    # Слой чистого металла
    base = nt.nodes.new("ShaderNodeBsdfPrincipled")
    base.location = (0, 100)
    base.inputs["Metallic"].default_value = 1.0
    base.inputs["Roughness"].default_value = np.clip(np.random.uniform(0.15, 0.45), 0.1, 0.7)
    base.inputs["Base Color"].default_value = (0.75,0.75,0.75,1)

    # Слой загрязнения (диэлектрик)
    dirt = nt.nodes.new("ShaderNodeBsdfPrincipled")
    dirt.location = (0, -100)
    dirt.inputs["Metallic"].default_value = 0.0
    dirt.inputs["Roughness"].default_value = np.clip(np.random.uniform(0.6, 0.95), 0.5, 0.98)
    dirt.inputs["Base Color"].default_value = (0.10,0.09,0.08,1)

    # Маска загрязнения = AO * Pointiness * Noise
    geom = _geometry(nt, (-1200, -100))
    ao = _ao(nt, (-1200, -300))
    n = _noise(nt, scale=np.random.uniform(3,8), detail=8, rough=0.6, loc=(-1000, -200))
    mul1 = nt.nodes.new("ShaderNodeMath"); mul1.operation = 'MULTIPLY'; mul1.location = (-800, -200)
    mul2 = nt.nodes.new("ShaderNodeMath"); mul2.operation = 'MULTIPLY'; mul2.location = (-600, -200)
    nt.links.new(geom.outputs["Pointiness"], mul1.inputs[0])
    nt.links.new(ao.outputs["Color"],        mul1.inputs[1])
    nt.links.new(mul1.outputs["Value"],      mul2.inputs[0])
    nt.links.new(n.outputs["Fac"],           mul2.inputs[1])
    ramp = _ramp(nt, (-400, -200), p0=0.3, p1=0.8)

    nt.links.new(mul2.outputs["Value"], ramp.inputs["Fac"])

    mix = _mix_shader(nt, (220, 0))
    nt.links.new(base.outputs["BSDF"], mix.inputs[1])
    nt.links.new(dirt.outputs["BSDF"], mix.inputs[2])
    nt.links.new(ramp.outputs["Color"], mix.inputs["Fac"])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    # Микрорельеф на итог
    bump = _bump(nt, strength=np.random.uniform(0.03, 0.10), loc=(50, -260))
    nt.links.new(n.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], mix.inputs[1].links[0].from_node.inputs["Normal"])
    return bproc.types.Material(mat) if BPROC else mat

def make_rusty_metal(name="Metal_Rust"):
    mat, nt, bsdf, out = _new_mat(name)
    # Чистый металл
    m = nt.nodes.new("ShaderNodeBsdfPrincipled")
    m.location = (0, 150)
    m.inputs["Metallic"].default_value = 1.0
    m.inputs["Roughness"].default_value = np.clip(np.random.uniform(0.2, 0.5), 0.15, 0.7)
    m.inputs["Base Color"].default_value = (0.72,0.72,0.72,1)

    # Ржавчина (диэлектрик)
    r = nt.nodes.new("ShaderNodeBsdfPrincipled")
    r.location = (0, -100)
    r.inputs["Metallic"].default_value = 0.0
    r.inputs["Specular"].default_value = 0.25
    r.inputs["Roughness"].default_value = np.clip(np.random.uniform(0.6, 0.95), 0.55, 0.98)
    rust_color = random.choice([(0.35,0.16,0.06,1), (0.45,0.18,0.05,1), (0.30,0.14,0.07,1)])
    r.inputs["Base Color"].default_value = rust_color

    # Маска ржавчины = AO * Noise * Musgrave (кустистая структура)
    ao = _ao(nt, (-1200, -250))
    n = _noise(nt, scale=np.random.uniform(2.0,4.0), detail=10, rough=0.5, loc=(-1050, -100))
    mus = _musgrave(nt, scale=np.random.uniform(5,12), detail=8, dimension=0.9, loc=(-1050, -320))
    mul1 = nt.nodes.new("ShaderNodeMath"); mul1.operation = 'MULTIPLY'; mul1.location = (-850, -180)
    mul2 = nt.nodes.new("ShaderNodeMath"); mul2.operation = 'MULTIPLY'; mul2.location = (-650, -180)
    nt.links.new(ao.outputs["Color"], mul1.inputs[0])
    nt.links.new(n.outputs["Fac"],   mul1.inputs[1])
    nt.links.new(mul1.outputs["Value"], mul2.inputs[0])
    nt.links.new(mus.outputs["Fac"], mul2.inputs[1])
    ramp = _ramp(nt, (-430, -180), p0=0.35, p1=0.8)
    nt.links.new(mul2.outputs["Value"], ramp.inputs["Fac"])

    mix = _mix_shader(nt, (220, 0))
    nt.links.new(m.outputs["BSDF"], mix.inputs[1])
    nt.links.new(r.outputs["BSDF"], mix.inputs[2])
    nt.links.new(ramp.outputs["Color"], mix.inputs["Fac"])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    # Бугристость ржавчины
    bump = _bump(nt, strength=np.random.uniform(0.05, 0.15), loc=(-200, -350))
    nt.links.new(mus.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], r.inputs["Normal"])
    return bproc.types.Material(mat) if BPROC else mat

def make_steel(name="Steel"):
    mat, nt, bsdf, out = _new_mat(name)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Specular"].default_value = 0.5
    bsdf.inputs["Roughness"].default_value = float(np.clip(np.random.uniform(0.12, 0.32), 0.06, 0.7))

    # Типичные тона стали (слегка холодные)
    col = random.choice([
        (0.62, 0.64, 0.66),
        (0.55, 0.57, 0.60),
        (0.68, 0.70, 0.72),
    ])
    bsdf.inputs["Base Color"].default_value = (*col, 1.0)

    tex, mapn = _texcoord_mapping(nt, (-1200, 0))
    n = _noise(nt,
               scale=np.random.uniform(10, 18),
               detail=np.random.uniform(4, 8),
               rough=np.random.uniform(0.35, 0.65),
               loc=(-950, 0))
    nt.links.new(mapn.outputs["Vector"], n.inputs["Vector"])

    ramp = _ramp(nt, (-750, 0), p0=0.35, p1=0.65)
    nt.links.new(n.outputs["Fac"], ramp.inputs["Fac"])

    bump = _bump(nt, strength=np.random.uniform(0.02, 0.08), dist=1.0, loc=(-450, -100))
    nt.links.new(ramp.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return bproc.types.Material(mat) if BPROC else mat


def make_stainless_steel(name="Stainless_Steel"):
    mat, nt, bsdf, out = _new_mat(name)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Specular"].default_value = 0.5
    bsdf.inputs["Roughness"].default_value = float(np.clip(np.random.uniform(0.06, 0.22), 0.02, 0.6))
    # Нержавейка — чуть ярче и холоднее
    col = random.choice([
        (0.72, 0.74, 0.76),
        (0.68, 0.71, 0.74),
        (0.76, 0.78, 0.80),
    ])
    bsdf.inputs["Base Color"].default_value = (*col, 1.0)

    # Микро-царапины: wave + noise в bump (очень тонко)
    tex, mapn = _texcoord_mapping(nt, (-1200, 0))

    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.location = (-950, 80)
    wave.wave_type = 'BANDS'
    wave.bands_direction = 'X'
    wave.inputs["Scale"].default_value = float(np.random.uniform(200, 600))
    wave.inputs["Distortion"].default_value = float(np.random.uniform(0.0, 2.0))
    nt.links.new(mapn.outputs["Vector"], wave.inputs["Vector"])

    n = _noise(nt, scale=np.random.uniform(12, 20), detail=4, rough=0.4, loc=(-950, -120))
    nt.links.new(mapn.outputs["Vector"], n.inputs["Vector"])

    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = 'MULTIPLY'; mul.location = (-720, -20)
    mul.inputs[1].default_value = 0.6
    nt.links.new(wave.outputs["Fac"], mul.inputs[0])

    add = nt.nodes.new("ShaderNodeMath"); add.operation = 'ADD'; add.location = (-560, -20)
    add.use_clamp = True
    nt.links.new(mul.outputs["Value"], add.inputs[0])
    nt.links.new(n.outputs["Fac"], add.inputs[1])

    ramp = _ramp(nt, (-380, -20), p0=0.45, p1=0.55)
    nt.links.new(add.outputs["Value"], ramp.inputs["Fac"])

    bump = _bump(nt, strength=np.random.uniform(0.01, 0.05), dist=1.0, loc=(-200, -160))
    nt.links.new(ramp.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Лёгкая анизотропия (без UV)
    if "Anisotropic" in bsdf.inputs:
        bsdf.inputs["Anisotropic"].default_value = float(np.random.uniform(0.05, 0.25))

    return bproc.types.Material(mat) if BPROC else mat


def make_brushed_steel(name="Brushed_Steel"):
    mat, nt, bsdf, out = _new_mat(name)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Specular"].default_value = 0.5

    base_rough = float(np.clip(np.random.uniform(0.12, 0.30), 0.05, 0.7))

    # Серо-холодная сталь
    col = random.choice([
        (0.62, 0.65, 0.68),
        (0.58, 0.61, 0.64),
        (0.68, 0.70, 0.72),
    ])
    bsdf.inputs["Base Color"].default_value = (*col, 1.0)

    # Анизотропия + направление по умолчанию (без Tangent)
    if "Anisotropic" in bsdf.inputs:
        bsdf.inputs["Anisotropic"].default_value = float(np.random.uniform(0.6, 0.9))
    if "Anisotropic Rotation" in bsdf.inputs:
        bsdf.inputs["Anisotropic Rotation"].default_value = float(np.random.uniform(0.0, 1.0))

    tex, mapn = _texcoord_mapping(nt, (-1200, 0))

    # Полосы шлифовки: Wave(BANDS) вдоль оси + небольшой шум
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.location = (-950, 60)
    wave.wave_type = 'BANDS'
    wave.bands_direction = random.choice(['X', 'Y'])
    wave.inputs["Scale"].default_value = float(np.random.uniform(250, 900))
    wave.inputs["Distortion"].default_value = float(np.random.uniform(0.2, 3.0))
    nt.links.new(mapn.outputs["Vector"], wave.inputs["Vector"])

    n = _noise(nt, scale=np.random.uniform(10, 18), detail=4, rough=0.45, loc=(-950, -140))
    nt.links.new(mapn.outputs["Vector"], n.inputs["Vector"])

    # Усилить полосы и подмешать шум
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = 'MULTIPLY'; mul.location = (-720, -20)
    mul.inputs[1].default_value = float(np.random.uniform(0.8, 1.2))
    nt.links.new(wave.outputs["Fac"], mul.inputs[0])

    add = nt.nodes.new("ShaderNodeMath"); add.operation = 'ADD'; add.location = (-560, -20)
    add.use_clamp = True
    nt.links.new(mul.outputs["Value"], add.inputs[0])

    nscale = nt.nodes.new("ShaderNodeMath"); nscale.operation='MULTIPLY'; nscale.location=(-720, -200)
    nscale.inputs[1].default_value = float(np.random.uniform(0.15, 0.35))
    nt.links.new(n.outputs["Fac"], nscale.inputs[0])
    nt.links.new(nscale.outputs["Value"], add.inputs[1])

    ramp = _ramp(nt, (-380, -20), p0=0.40, p1=0.60)
    nt.links.new(add.outputs["Value"], ramp.inputs["Fac"])

    bump = _bump(nt, strength=np.random.uniform(0.02, 0.09), dist=1.0, loc=(-200, -160))
    nt.links.new(ramp.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Roughness = base + небольшая модуляция от полос
    val = nt.nodes.new("ShaderNodeValue"); val.location = (-560, 180)
    val.outputs[0].default_value = base_rough

    amp = nt.nodes.new("ShaderNodeMath"); amp.operation='MULTIPLY'; amp.location=(-380, 180)
    amp.inputs[1].default_value = float(np.random.uniform(0.05, 0.14))
    nt.links.new(wave.outputs["Fac"], amp.inputs[0])

    radd = nt.nodes.new("ShaderNodeMath"); radd.operation='ADD'; radd.location=(-200, 180)
    radd.use_clamp = True
    nt.links.new(val.outputs[0], radd.inputs[0])
    nt.links.new(amp.outputs["Value"], radd.inputs[1])

    nt.links.new(radd.outputs["Value"], bsdf.inputs["Roughness"])

    return bproc.types.Material(mat) if BPROC else mat


def make_galvanized_steel(name="Galvanized_Steel"):
    mat, nt, bsdf, out = _new_mat(name)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Specular"].default_value = 0.5

    base_rough = float(np.clip(np.random.uniform(0.25, 0.55), 0.08, 0.9))
    bsdf.inputs["Roughness"].default_value = base_rough

    # База — светло-серая
    base_col = (0.70, 0.72, 0.74, 1.0)

    tex, mapn = _texcoord_mapping(nt, (-1200, 0))

    # "Спанглы" цинка: Voronoi + Noise
    v = _voronoi(nt, scale=np.random.uniform(25, 60), loc=(-950, 40))
    v.feature = 'F1'
    nt.links.new(mapn.outputs["Vector"], v.inputs["Vector"])

    n = _noise(nt, scale=np.random.uniform(6, 12), detail=5, rough=0.5, loc=(-950, -180))
    nt.links.new(mapn.outputs["Vector"], n.inputs["Vector"])

    # Маска спанглов: Fac(Voronoi) -> Ramp
    ramp = _ramp(nt, (-720, 40), p0=0.25, p1=0.75,
                 c0=(0.0,0.0,0.0,1.0),
                 c1=(1.0,1.0,1.0,1.0))
    nt.links.new(v.outputs["Distance"], ramp.inputs["Fac"])

    # Цветовые вариации: Mix base_col с чуть более тёмным/светлым по ramp*noise
    rgb_base = nt.nodes.new("ShaderNodeRGB"); rgb_base.location = (-720, 240)
    rgb_base.outputs["Color"].default_value = base_col

    rgb_var = nt.nodes.new("ShaderNodeRGB"); rgb_var.location = (-720, 120)
    # слегка тонированная вариация
    var_col = random.choice([
        (0.62, 0.64, 0.66, 1.0),
        (0.76, 0.78, 0.80, 1.0),
        (0.66, 0.69, 0.72, 1.0),
    ])
    rgb_var.outputs["Color"].default_value = var_col

    # fac = ramp * (0.6 + 0.4*noise)
    nmap = nt.nodes.new("ShaderNodeMath"); nmap.operation='MULTIPLY'; nmap.location=(-560, -180)
    nmap.inputs[1].default_value = 0.4
    nt.links.new(n.outputs["Fac"], nmap.inputs[0])

    nadd = nt.nodes.new("ShaderNodeMath"); nadd.operation='ADD'; nadd.location=(-380, -180)
    nadd.inputs[0].default_value = 0.6
    nadd.use_clamp = True
    nt.links.new(nmap.outputs["Value"], nadd.inputs[1])

    facmul = nt.nodes.new("ShaderNodeMath"); facmul.operation='MULTIPLY'; facmul.location=(-560, 20)
    nt.links.new(ramp.outputs["Color"], facmul.inputs[0])  # RGB -> float автоматически возьмёт канал
    nt.links.new(nadd.outputs["Value"], facmul.inputs[1])

    mixc = nt.nodes.new("ShaderNodeMixRGB"); mixc.location = (-380, 200)
    mixc.blend_type = 'MIX'
    nt.links.new(facmul.outputs["Value"], mixc.inputs["Fac"])
    nt.links.new(rgb_base.outputs["Color"], mixc.inputs[1])
    nt.links.new(rgb_var.outputs["Color"], mixc.inputs[2])
    nt.links.new(mixc.outputs["Color"], bsdf.inputs["Base Color"])

    # Бамп по спанглам
    bump = _bump(nt, strength=np.random.uniform(0.03, 0.10), dist=1.0, loc=(-200, -60))
    nt.links.new(ramp.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Roughness чуть пляшет по ramp
    val = nt.nodes.new("ShaderNodeValue"); val.location = (-560, 380)
    val.outputs[0].default_value = base_rough
    amp = nt.nodes.new("ShaderNodeMath"); amp.operation='MULTIPLY'; amp.location=(-380, 380)
    amp.inputs[1].default_value = float(np.random.uniform(0.05, 0.18))
    nt.links.new(ramp.outputs["Color"], amp.inputs[0])
    add = nt.nodes.new("ShaderNodeMath"); add.operation='ADD'; add.location=(-200, 380)
    add.use_clamp = True
    nt.links.new(val.outputs[0], add.inputs[0])
    nt.links.new(amp.outputs["Value"], add.inputs[1])
    nt.links.new(add.outputs["Value"], bsdf.inputs["Roughness"])

    return bproc.types.Material(mat) if BPROC else mat


def make_blackened_steel(name="Blackened_Steel"):
    mat, nt, bsdf, out = _new_mat(name)

    # Металлическая база
    metal = nt.nodes.new("ShaderNodeBsdfPrincipled")
    metal.location = (0, 120)
    metal.inputs["Metallic"].default_value = 1.0
    metal.inputs["Roughness"].default_value = float(np.clip(np.random.uniform(0.25, 0.55), 0.08, 0.95))
    metal.inputs["Base Color"].default_value = random.choice([
        (0.10, 0.10, 0.11, 1.0),
        (0.13, 0.13, 0.14, 1.0),
        (0.16, 0.16, 0.17, 1.0),
    ])

    # Оксидная плёнка (диэлектрик), более матовая
    oxide = nt.nodes.new("ShaderNodeBsdfPrincipled")
    oxide.location = (0, -80)
    oxide.inputs["Metallic"].default_value = 0.0
    oxide.inputs["Specular"].default_value = 0.25
    oxide.inputs["Roughness"].default_value = float(np.clip(np.random.uniform(0.6, 0.95), 0.35, 0.98))
    oxide.inputs["Base Color"].default_value = random.choice([
        (0.03, 0.03, 0.03, 1.0),
        (0.05, 0.05, 0.05, 1.0),
        (0.07, 0.07, 0.08, 1.0),
    ])

    # Маска: AO * Pointiness * Noise (плёнка сильнее в углублениях)
    geom = _geometry(nt, (-1200, -80))
    ao = _ao(nt, (-1200, -280))
    tex, mapn = _texcoord_mapping(nt, (-1200, 80))
    n = _noise(nt, scale=np.random.uniform(3, 7), detail=6, rough=0.6, loc=(-980, -160))
    nt.links.new(mapn.outputs["Vector"], n.inputs["Vector"])

    mul1 = nt.nodes.new("ShaderNodeMath"); mul1.operation='MULTIPLY'; mul1.location=(-800, -180)
    mul2 = nt.nodes.new("ShaderNodeMath"); mul2.operation='MULTIPLY'; mul2.location=(-600, -180)
    nt.links.new(geom.outputs["Pointiness"], mul1.inputs[0])
    nt.links.new(ao.outputs["Color"],        mul1.inputs[1])
    nt.links.new(mul1.outputs["Value"],      mul2.inputs[0])
    nt.links.new(n.outputs["Fac"],           mul2.inputs[1])

    ramp = _ramp(nt, (-420, -180), p0=0.25, p1=0.8)
    nt.links.new(mul2.outputs["Value"], ramp.inputs["Fac"])

    mix = _mix_shader(nt, (220, 20))
    nt.links.new(metal.outputs["BSDF"], mix.inputs[1])
    nt.links.new(oxide.outputs["BSDF"], mix.inputs[2])
    nt.links.new(ramp.outputs["Color"], mix.inputs["Fac"])
    nt.links.new(mix.outputs["Shader"], out.inputs["Surface"])

    # Микро-рельеф по шуму
    bump = _bump(nt, strength=np.random.uniform(0.02, 0.08), dist=1.0, loc=(-200, -320))
    nt.links.new(n.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], metal.inputs["Normal"])
    nt.links.new(bump.outputs["Normal"], oxide.inputs["Normal"])

    return bproc.types.Material(mat) if BPROC else mat

def make_cast_iron(name="Cast_Iron"):
    mat, nt, bsdf, out = _new_mat(name)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Specular"].default_value = 0.5
    bsdf.inputs["Roughness"].default_value = float(np.clip(np.random.uniform(0.45, 0.85), 0.2, 0.98))

    # Чугун — темнее (почти чёрный графитовый)
    col = random.choice([
        (0.10, 0.10, 0.11),
        (0.13, 0.13, 0.14),
        (0.16, 0.16, 0.17),
    ])
    bsdf.inputs["Base Color"].default_value = (*col, 1.0)

    tex, mapn = _texcoord_mapping(nt, (-1200, 0))

    # Питтинг/шероховатость: Musgrave + Noise
    mus = _musgrave(nt, scale=np.random.uniform(18, 40), detail=10, dimension=0.8, loc=(-950, -40))
    nt.links.new(mapn.outputs["Vector"], mus.inputs["Vector"])

    n = _noise(nt, scale=np.random.uniform(6, 12), detail=6, rough=0.6, loc=(-950, -240))
    nt.links.new(mapn.outputs["Vector"], n.inputs["Vector"])

    mul = nt.nodes.new("ShaderNodeMath"); mul.operation='MULTIPLY'; mul.location=(-720, -140)
    mul.inputs[1].default_value = float(np.random.uniform(0.6, 1.2))
    nt.links.new(mus.outputs["Fac"], mul.inputs[0])

    add = nt.nodes.new("ShaderNodeMath"); add.operation='ADD'; add.location=(-560, -140)
    add.use_clamp = True
    nt.links.new(mul.outputs["Value"], add.inputs[0])

    nscale = nt.nodes.new("ShaderNodeMath"); nscale.operation='MULTIPLY'; nscale.location=(-720, -300)
    nscale.inputs[1].default_value = float(np.random.uniform(0.25, 0.5))
    nt.links.new(n.outputs["Fac"], nscale.inputs[0])
    nt.links.new(nscale.outputs["Value"], add.inputs[1])

    ramp = _ramp(nt, (-380, -140), p0=0.25, p1=0.75)
    nt.links.new(add.outputs["Value"], ramp.inputs["Fac"])

    bump = _bump(nt, strength=np.random.uniform(0.07, 0.20), dist=1.0, loc=(-200, -280))
    nt.links.new(ramp.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    return bproc.types.Material(mat) if BPROC else mat

def make_plastic_new(name="Plastic_New"):
    mat, nt, bsdf, out = _new_mat(name)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Specular"].default_value = 0.5
    bsdf.inputs["Roughness"].default_value = np.clip(np.random.uniform(0.15, 0.35), 0.1, 0.5)
    bsdf.inputs["Clearcoat"].default_value = np.random.uniform(0.0, 0.15)
    col = _color()  # нейтральная палитра
    # Лёгкая вариация тона
    hsv = _hsv(nt, (-100, 200), h=np.random.uniform(-0.02, 0.02), s=np.random.uniform(0.9,1.1), v=np.random.uniform(0.95,1.05))
    rgb = nt.nodes.new("ShaderNodeRGB")
    rgb.location = (-300, 200)
    rgb.outputs["Color"].default_value = col
    nt.links.new(rgb.outputs["Color"], hsv.inputs["Color"])
    nt.links.new(hsv.outputs["Color"], bsdf.inputs["Base Color"])
    # Микрошум
    n = _noise(nt, scale=np.random.uniform(20, 40), detail=3, rough=0.35, loc=(-400, -120))
    ramp = _ramp(nt, (-200, -120), p0=0.45, p1=0.55)
    nt.links.new(n.outputs["Fac"], ramp.inputs["Fac"])
    bump = _bump(nt, strength=np.random.uniform(0.01, 0.04), dist=1.0, loc=(0, -200))
    nt.links.new(ramp.outputs["Color"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return bproc.types.Material(mat) if BPROC else mat

def make_plastic_old(name="Plastic_Old"):
    mat, nt, bsdf, out = _new_mat(name)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Specular"].default_value = 0.45
    bsdf.inputs["Roughness"].default_value = np.clip(np.random.uniform(0.45, 0.8), 0.35, 0.9)
    # База цвета
    rgb = nt.nodes.new("ShaderNodeRGB")
    rgb.location = (-600, 200)
    rgb.outputs["Color"].default_value = _color()
    nt.links.new(rgb.outputs["Color"], bsdf.inputs["Base Color"])
    # Грязь/потёртости: mix по маске (Noise*Pointiness)
    dirt_col = nt.nodes.new("ShaderNodeRGB")
    dirt_col.location = (-600, 0)
    dirt_col.outputs["Color"].default_value = (0.12,0.11,0.10,1)
    mix_col = nt.nodes.new("ShaderNodeMixRGB")
    mix_col.location = (-200, 120)
    mix_col.blend_type = 'MIX'
    mix_col.inputs["Fac"].default_value = 0.0
    nt.links.new(rgb.outputs["Color"],     mix_col.inputs[1])
    nt.links.new(dirt_col.outputs["Color"], mix_col.inputs[2])
    # Маска загрязнения
    geom = _geometry(nt, (-1200, 100))
    n = _noise(nt, scale=np.random.uniform(3,8), detail=6, rough=0.6, loc=(-1000, 0))
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation='MULTIPLY'; mul.location=(-800, 50)
    nt.links.new(geom.outputs["Pointiness"], mul.inputs[0])
    nt.links.new(n.outputs["Fac"],          mul.inputs[1])
    ramp = _ramp(nt, (-600, 50), p0=0.4, p1=0.85)
    nt.links.new(mul.outputs["Value"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], mix_col.inputs["Fac"])
    nt.links.new(mix_col.outputs["Color"], bsdf.inputs["Base Color"])
    # Микрорельеф сильнее
    mus = _musgrave(nt, scale=np.random.uniform(8,16), detail=8, dimension=1.0, loc=(-600, -150))
    bump = _bump(nt, strength=np.random.uniform(0.03, 0.10), loc=(-350, -150))
    nt.links.new(mus.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return bproc.types.Material(mat) if BPROC else mat

# ---------------------- РОУТЕР ПО СТИЛЯМ ----------------------------

def make_material(style: str, name: str = None):
    style = (style or "").lower()
    name = name or f"Mat_{style}"

    if style in ("metal", "polished_metal"):
        return make_metal(name)
    if style in ("dirty_metal", "metal_dirty"):
        return make_dirty_metal(name)
    if style in ("rust", "rusty_metal"):
        return make_rusty_metal(name)

    if style in ("steel", "mild_steel"):
        return make_steel(name)
    if style in ("stainless", "stainless_steel", "inox"):
        return make_stainless_steel(name)
    if style in ("brushed", "brushed_steel", "satin_steel"):
        return make_brushed_steel(name)
    if style in ("galvanized", "galvanized_steel", "zinc_coated"):
        return make_galvanized_steel(name)
    if style in ("blackened_steel", "black_oxide", "blued_steel"):
        return make_blackened_steel(name)

    if style in ("cast_iron", "iron_cast", "ductile_iron"):
        return make_cast_iron(name)

    if style in ("plastic_new", "new_plastic"):
        return make_plastic_new(name)
    if style in ("plastic_old", "old_plastic", "worn_plastic"):
        return make_plastic_old(name)

    return make_metal(name)




def make_random_material(allowed: Optional[Iterable[str]] = None,
                         weights: Optional[Iterable[float]] = None,
                         name_prefix: str = "MatRand",
                         seed: Optional[int] = None) -> Tuple[object, str]:
    """
    Возвращает (material, style), где material — bpy.types.Material или bproc.types.Material.
    allowed — подмножество стилей; если None, берётся весь пул MATERIAL_STYLES.
    weights — вероятности выбора (та же длина, что и allowed).
    name_prefix — префикс имени материала.
    seed — фиксирует случайность для воспроизводимости.
    """
    styles = tuple(allowed) if allowed else MATERIAL_STYLES
    assert len(styles) > 0, "Список стилей пуст."
    if seed is not None:
        rnd = random.Random(seed)
        choice = rnd.choices(styles, weights=weights, k=1)[0]
    else:
        choice = random.choices(styles, weights=weights, k=1)[0]
    mat_name = f"{name_prefix}_{choice}_{random.randint(1000, 9999)}"
    mat = make_material(choice, mat_name)
    return mat, choice

#creating new material for object
def create_mat(mat_name):
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True

    node_tree = mat.node_tree
    nodes = node_tree.nodes

    bsdf = nodes.get("Principled BSDF") 

    assert(bsdf) # make sure it exists to continue

    #color ramp -> bsdf(metallic)
    cr_m = nodes.new(type="ShaderNodeValToRGB")
    cr_m.location = (-450, 400)
    cr_m.color_ramp.elements[0].position = (np.random.uniform(0.2, 0.7))
    cr_m.color_ramp.elements[1].position = (np.random.uniform(0.8, 1))
    node_tree.links.new(cr_m.outputs[0], bsdf.inputs[4])

    #Nois texture -> color ramp 
    nt_cr = nodes.new(type="ShaderNodeTexNoise")
    nt_cr.location = (-600, 150)
    nt_cr.inputs["Scale"].default_value = 4
    nt_cr.inputs["Detail"].default_value = np.random.uniform(7, 20)
    nt_cr.inputs["Roughness"].default_value = np.random.uniform(0.3, 0.8)
    #color ramp -> bsdf(Roughness)
    cr_r = nodes.new(type="ShaderNodeValToRGB")
    cr_r.location = (-300, 150)
    cr_r.color_ramp.elements[0].position = (np.random.uniform(0.2, 0.7))
    cr_r.color_ramp.elements[1].position = (np.random.uniform(0.9, 1))
    #noise textur for color ramp
    node_tree.links.new(nt_cr.outputs[1], cr_r.inputs[0])
    #color ramp -> bsdf(roughness)
    node_tree.links.new(cr_r.outputs[0], bsdf.inputs[7])

    #create node bump -> bsdf(normal)
    bump_node = nodes.new(type="ShaderNodeBump")
    bump_node.location = (-200, -100)
    bump_node.inputs["Strength"].default_value = np.random.uniform(0.01, 0.3)
    node_tree.links.new(bump_node.outputs[0], bsdf.inputs[20])

    #create node color ramp -> bump
    cr_b = nodes.new(type="ShaderNodeValToRGB")
    cr_b.location = (-500, -100)
    cr_b.color_ramp.elements[0].position = (np.random.uniform(0.4, 0.8))
    cr_b.color_ramp.elements[1].position = (np.random.uniform(0.9, 1))
    node_tree.links.new(cr_b.outputs[0], bump_node.inputs[2])

    #create node Voronoi Texture -> color ramp 
    voronoi_node = nodes.new(type="ShaderNodeTexVoronoi")
    voronoi_node.location = (-700, -100)
    node_tree.links.new(voronoi_node.outputs[1], cr_b.inputs[0])

    #create node mapping -> Voronoi Texture
    mapping_node = nodes.new(type="ShaderNodeMapping")
    mapping_node.location = (-900, -50)
    node_tree.links.new(mapping_node.outputs[0], voronoi_node.inputs[0])

    #create node noise texture -> mapping 
    nt_mapping = nodes.new(type="ShaderNodeTexNoise")
    nt_mapping.location = (-1100, 0)
    nt_mapping.inputs["Scale"].default_value = 4
    nt_mapping.inputs["Detail"].default_value = np.random.uniform(15, 30)
    nt_mapping.inputs["Roughness"].default_value = np.random.uniform(0.5, 0.8)
    node_tree.links.new(nt_mapping.outputs[1], mapping_node.inputs[0])

    #create node texture coordinate -> noise texture
    texcoord_node = nodes.new(type="ShaderNodeTexCoord")
    texcoord_node.location = (-1300, 0)
    node_tree.links.new(texcoord_node.outputs[3], nt_mapping.inputs[0])
    
    #creation base color for object
    a = 1
    i = [0, 1, 2, 3]
    rand = (random.sample(i, 1))[0]
    #print(rand)
    if rand == 0:
        r = 0.322
        g = np.random.uniform(0.09, 0.22)
        b = np.random.uniform(0, 0.03)
    elif rand == 1: 
        r = 0.542
        g = np.random.uniform(0.08, 0.35)
        b = np.random.uniform(0, 0.06)
    elif rand == 2:
        grey_col = np.random.uniform(0.08, 0.4)
        r = grey_col
        g = grey_col
        b = grey_col
    else:
        grey_col = np.random.uniform(0.08, 0.2)
        r = grey_col
        g = grey_col
        b = grey_col    

    rgb_color = [r, g, b]
    print("color: {}".format(rgb_color))
    bsdf.inputs["Base Color"].default_value = [rgb_color[0], rgb_color[1], rgb_color[2], a]
    return bproc.types.Material(mat)


def custom_load_CCmaterials(folder_path: str = "resources/custom_metal_textures", 
                     used_assets: Optional[List[str]] = None, 
                     preload: bool = False,
                     fill_used_empty_materials: bool = False, 
                     add_custom_properties: Optional[Dict[str, str]] = None) -> List[bpy.types.Material]:
    """
    Загружает металлические текстуры 8K с сайта ambientCG.com и создает из них материалы в Blender.

    Args:
        folder_path (str): Путь к папке с металлическими текстурами. По умолчанию "resources/custom_metal_textures".
        used_assets (Optional[List[str]]): Список названий металлических активов для загрузки. Если None, загружаются все доступные активы.
        preload (bool): Если True, загружаются только названия материалов без полной загрузки текстур. По умолчанию False.
        fill_used_empty_materials (bool): Если True, полностью загружаются предварительно загруженные материалы. По умолчанию False.
        add_custom_properties (Optional[Dict[str, str]]): Словарь для добавления кастомных свойств в материалы. По умолчанию None.

    Returns:
        List[bpy.types.Material]: Список созданных материалов Blender.

    Raises:
        FileNotFoundError: Если указанный путь к папке не существует.
        Exception: Если одновременно установлены preload и fill_used_empty_materials.
    """
    folder_path = resolve_path(folder_path)

    if add_custom_properties is None:
        add_custom_properties = {}

    if preload and fill_used_empty_materials:
        raise Exception("Preload and fill used empty materials can not be done at the same time, check config!")

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        materials = []
        for asset_dir in os.listdir(folder_path):
            asset_path = os.path.join(folder_path, asset_dir)

            if not os.path.isdir(asset_path):
                continue

            if used_assets and asset_dir not in used_assets:
                continue

            # Структура имени файла
            base_image_path = os.path.join(asset_path, f"{asset_dir}_Color.jpg")
            if not os.path.exists(base_image_path):
                continue

            # Остальные пути к картам текстур
            ambient_occlusion_image_path = os.path.join(asset_path, f"{asset_dir}_AmbientOcclusion.jpg")  # Если есть
            metallic_image_path = os.path.join(asset_path, f"{asset_dir}_Metalness.jpg")
            roughness_image_path = os.path.join(asset_path, f"{asset_dir}_Roughness.jpg")
            alpha_image_path = os.path.join(asset_path, f"{asset_dir}_Opacity.jpg")  # Если есть
            normal_image_path = os.path.join(asset_path, f"{asset_dir}_NormalDX.jpg")  # Или NormalGL, если нужно
            displacement_image_path = os.path.join(asset_path, f"{asset_dir}_Displacement.jpg")

            if fill_used_empty_materials:
                new_mat = MaterialLoaderUtility.find_cc_material_by_name(asset_dir, add_custom_properties)
            else:
                new_mat = MaterialLoaderUtility.create_new_cc_material(asset_dir, add_custom_properties)

            if preload:
                # Упрощенный код для случая предзагрузки
                materials.append(Material(new_mat))
                continue

            if fill_used_empty_materials and not MaterialLoaderUtility.is_material_used(new_mat):
                continue

            _CCMaterialLoader.create_material(new_mat, base_image_path, ambient_occlusion_image_path,
                                              metallic_image_path, roughness_image_path, alpha_image_path,
                                              normal_image_path, displacement_image_path)

            materials.append(Material(new_mat))
        return materials
    else:
        raise FileNotFoundError(f"The folder path does not exist: {folder_path}")



