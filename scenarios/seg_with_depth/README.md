

Поле images[]

```json
{
  "id": 12,
  "file_name": "images/000012.jpg",
  "width": 1280,
  "height": 720,

  "depth_raw_file": "depth_raw/000012.png",
  "depth_inpainted_file": "depth_inpainted/000012.png",
  "depth_unit": "mm",
  "depth_scale": 0.001,

  "instances_mask_file": "masks/000012.png",
  "valid_depth_mask_file": "mask_valid/000012.png",

  "camera_id": 0,
  "scene_id": 3,
  "frame_id": 12
}
```

Поле annotations[]

```json
{
  "id": 9001,
  "image_id": 12,
  "category_id": 1, // obj_1 или obj_2 и тд
  "bbox": [x, y, w, h],                   
  "area": 12345,
  "iscrowd": 0,

  "segmentation": { "counts": "...", "size": [H, W] },
  "instance_index": 3,

  "fragment_id": 5, // 1..5 (внутри объекта/кадра)
  "fracture_method": "voronoi",
  "fracture_seed": 1337,
}
```

Если нужен “чистый COCO” → используем segmentation в annotation[]

Если нужен пайплайн/дебаг → используем instances_mask_file в images[]

`fragment_id` - локальный индекс на кадр

Пояснение полям images[]

`depth_inpainted_file` - “идеальная/чистая” глубина из рендера без артефактов сенсора.
`depth_raw_file` - “как будто снято реальной RGB-D камерой” — глубина после симуляции деградаций.

!Важно: depth_raw может содержать 0 там, где depth_inpainted не 0.

`valid_depth_mask_file` - бинарная маска пикселей, где depth_raw считается валидной измеренной глубиной. То есть valid_depth_mask = (depth_raw > 0).

`instances_mask_file` - packed instance-id карта на кадр. Это не COCO-обязательная вещь, а ускоритель/удобство.

