无人机输电线路防外力破坏图像数据集（1500张，voc标签，塔吊钩机等目标）
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e6f6acaf0d5940e3a5501c55eebb0284.jpeg)


Annotations are the label 
images are the dataset
The Chinese for the category labeling in the image, corresponding to English:
yanhuo --- smoke
huoyan --- flame
diaoche --- crane
tadiao --- tower crane
daoxianyiwu --- antenna threats
shigongjixie --- construction machinery
daoxian --- antenna

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/11f25b8b730a469e9705347c54de9ab6.jpeg)

| 中文类别名（图像标注） | 英文对应名称          |
|----------------------|---------------------|
| 烟火                 | yanhuo              |
| 火焰                 | huoyan              |
| 吊车                 | diaoche             |
| 塔吊                 | tadiao              |
| 导线异物             | daoxianyiwu         |
| 施工机械             | shigongjixie        |
| 导线                 | daoxian             |

> 💡 说明：
> -  **1500 张图像**，采用 **PASCAL VOC 格式标注**（即每张图像对应一个 `.xml` 标注文件）。
> - 主要用于输电线路周边**外力破坏风险识别**，如施工机械入侵、火灾隐患、异物挂线等场景的智能监控与预警。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/77b0aaeab10d495b9d7d90dd3be3d015.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c07b01733cff44e6b8373c4eee8bd511.png)
1
针对 **输电线路防外力破坏图像数据集**（1500张，VOC格式标注）的完整 YOLOv8 训练代码，包含数据集转换、配置文件生成、模型训练与验证全流程。

---

### ✅ 一、数据集说明

- **图像数量**：1500 张（建议按 7:2:1 划分 train/val/test ≈ 1050 / 300 / 150）
- **标注格式**：PASCAL VOC（`.xml` 文件）
- **类别（7类）**：
  - yanhuo（烟火）
  - huoyan（火焰）
  - diaoche（吊车）
  - tadiao（塔吊）
  - daoxianyiwu（导线异物）
  - shigongjixie（施工机械）
  - daoxian（导线）

> ⚠️ 注意：YOLOv8 原生支持 YOLO 格式（`.txt`），需将 VOC 转换为 YOLO 格式。

---

### ✅ 二、目录结构建议

```
power_line_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

---

### ✅ 三、1. 将 VOC 标注转换为 YOLO 格式

```python
# voc2yolo.py
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

# 类别映射（顺序必须与 pld.yaml 中 names 一致）
classes = ['yanhuo', 'huoyan', 'diaoche', 'tadiao', 'daoxianyiwu', 'shigongjixie', 'daoxian']

def convert_voc_to_yolo(xml_file, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    yolo_lines = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text.strip()
        if cls_name not in classes:
            continue
        cls_id = classes.index(cls_name)

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # 归一化中心点和宽高
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / float(img_width)
        height = (ymax - ymin) / float(img_height)

        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return yolo_lines

def process_dataset(image_dir, ann_dir, output_label_dir):
    os.makedirs(output_label_dir, exist_ok=True)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))]
    for img_file in tqdm(image_files, desc=f"Processing {os.path.basename(image_dir)}"):
        img_name = os.path.splitext(img_file)[0]
        xml_path = os.path.join(ann_dir, img_name + '.xml')
        if not os.path.exists(xml_path):
            continue

        # 获取图像尺寸（可使用 PIL 或 OpenCV）
        from PIL import Image
        with Image.open(os.path.join(image_dir, img_file)) as im:
            w, h = im.size

        yolo_labels = convert_voc_to_yolo(xml_path, w, h)
        label_path = os.path.join(output_label_dir, img_name + '.txt')
        with open(label_path, 'w') as f:
            f.write("\n".join(yolo_labels))

# 使用示例（请根据实际路径修改）
if __name__ == "__main__":
    # 假设原始数据在 ./raw_data/images 和 ./raw_data/Annotations
    raw_img_dir = "./raw_data/images"
    raw_ann_dir = "./raw_data/Annotations"

    # 划分数据集（此处简化为全部用于训练，实际应划分）
    import shutil
    from sklearn.model_selection import train_test_split

    all_imgs = [f for f in os.listdir(raw_img_dir) if f.endswith(('.jpg', '.png'))]
    train_imgs, test_imgs = train_test_split(all_imgs, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(test_imgs, test_size=0.5, random_state=42)

    splits = {
        'train': train_imgs,
        'val': val_imgs,
        'test': test_imgs
    }

    for split, img_list in splits.items():
        img_dst = f"./power_line_dataset/images/{split}"
        lbl_dst = f"./power_line_dataset/labels/{split}"
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)

        for img in img_list:
            shutil.copy(os.path.join(raw_img_dir, img), os.path.join(img_dst, img))
        process_dataset(img_dst, raw_ann_dir, lbl_dst)
```

> 📌 运行前安装依赖：  
> ```bash
> pip install tqdm pillow scikit-learn
> ```

---

### ✅ 四、2. 创建数据集配置文件 `pld.yaml`

```yaml
# pld.yaml
path: ./power_line_dataset
train: images/train
val: images/val
test: images/test

nc: 7
names: ['yanhuo', 'huoyan', 'diaoche', 'tadiao', 'daoxianyiwu', 'shigongjixie', 'daoxian']
```

---

### ✅ 五、3. YOLOv8 训练代码（Python）

```python
# train_power_line.py
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')  # 可替换为 yolov8s.pt 等

# 开始训练
results = model.train(
    data='pld.yaml',
    epochs=150,
    imgsz=640,
    batch=16,                # 根据GPU调整（如 8/16/32）
    name='yolov8n_powerline',
    device=0,                # GPU ID
    patience=30,
    save=True,
    plots=True,
    exist_ok=False           # 若实验名已存在则报错
)

# 验证测试集性能
metrics = model.val(data='pld.yaml', split='test')
print(f"\n✅ 测试集结果:")
print(f"mAP@0.5:     {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95:{metrics.box.map:.4f}")
print(f"Precision:   {metrics.box.p:.4f}")
print(f"Recall:      {metrics.box.r:.4f}")
```

---

### ✅ 六、4. 命令行训练（等效）

```bash
yolo detect train \
    data=pld.yaml \
    model=yolov8n.pt \
    epochs=150 \
    imgsz=640 \
    batch=16 \
    name=yolov8n_powerline \
    device=0
```

---

### ✅ 七、注意事项

| 项目 | 说明 |
|------|------|
| **输入尺寸** | 建议保持 640×640，YOLOv8 自动处理非正方形图像 |
| **小目标检测** | 若“导线”“异物”等目标较小，可尝试 `imgsz=1280` 或使用 YOLOv8m/l |
| **数据增强** | YOLOv8 默认启用 Mosaic、HSV、翻转等，适合本场景 |
| **部署** | 训练后可用 `model.export(format='onnx')` 导出模型用于边缘设备 |

---

### ✅ 八、典型应用场景

- 输电线路通道 **施工机械入侵告警**
- 山火/焚烧秸秆引发的 **烟火火焰识别**
- 导线挂异物（风筝、塑料膜）**自动巡检**
- 电力巡检无人机/视频监控系统的 **AI智能分析模块**

---


