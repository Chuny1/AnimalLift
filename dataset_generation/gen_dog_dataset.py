import os
import json
import random
import argparse
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from animallift.config import parse_configured_args, require_file

BASE_DIR = JSON_PATH = TEXTURE_DIR = INPUT_TEXTURE_PATH = None

TOTAL_COUNT = 4000

########################################
# ========= Generation parameters =========
########################################

breeds = [
"阿富汗猎犬","艾尔谷梗","秋田犬","美国恶霸犬","美国爱斯基摩犬","美国猎狐犬",
"美国斯塔福郡梗","澳大利亚牧牛犬","澳大利亚牧羊犬","澳大利亚梗",
"巴仙吉犬","巴吉度猎犬","比格犬","比利时马里努阿犬","比利时特伏丹犬","比利时牧羊犬",
"伯恩山犬","比熊犬","黑褐猎浣熊犬","边境牧羊犬","边境梗","波尔多犬",
"波士顿梗","拳师犬","布列塔尼犬","布鲁塞尔格里芬犬","斗牛犬","牛头梗",
"凯恩梗","迦南犬","卡迪根威尔士柯基","骑士查理王小猎犬","切萨皮克湾寻回犬","吉娃娃",
"中华田园犬","中国冠毛犬","松狮犬","克伦伯猎犬","可卡犬",
"柯利犬","库瓦兹犬","腊肠犬","达尔马提亚犬","丹迪丁蒙梗","杜宾犬",
"阿根廷杜高犬","荷兰牧羊犬",
"英国斗牛犬","英国可卡犬","英国玩具梗","英国史宾格犬",
"田野小猎犬","芬兰猎犬","芬兰拉普猎犬","法国斗牛犬",
"德国牧羊犬","德国短毛指示犬","德国硬毛指示犬","德国猎梗",
"巨型雪纳瑞","迷你雪纳瑞","标准雪纳瑞",
"灵缇犬","哈威那犬","哈萨克猎犬","喜乐蒂牧羊犬",
"北海道犬","霍瓦特犬",
"伊比赞猎犬","冰岛牧羊犬","爱尔兰猎狼犬","爱尔兰雪达犬","爱尔兰水猎犬",
"爱尔兰梗","意大利灰狗",
"杰克罗素梗","日本尖嘴犬",
"凯利蓝梗","纪州犬","科蒙多犬","韩国珍岛犬",
"拉布拉多犬","拉戈托犬","湖畔梗","兰开夏赫勒犬",
"罗秦犬","罗威纳犬",
"马尔济斯犬","曼彻斯特梗","马雷马牧羊犬","马士提夫犬",
"迷你牛头梗","迷你杜宾犬",
"新斯科舍诱鸭寻回犬","纽芬兰犬","挪威猎鹿犬","挪威伦德猎犬",
"奥达猎犬","英国古代牧羊犬","奥地利黑褐猎犬",
"巴比特犬","巴仙吉犬","北京犬","秘鲁无毛犬",
"法老猎犬","普罗特猎犬","博美犬","贵宾犬",
"葡萄牙水犬","葡萄牙牧羊犬","巴哥犬",
"比利牛斯山犬","比利牛斯牧羊犬",
"罗得西亚背脊犬",
"柴犬","萨摩耶犬","沙皮犬","设得兰牧羊犬",
"西施犬","西伯利亚哈士奇","丝毛梗","斯凯梗","斯卢吉犬",
"史宾诺犬","圣伯纳犬","斯塔福郡斗牛梗",
"瑞典瓦汉德犬",
"泰迪犬（贵宾别称）","西藏梗","西藏猎犬",
"玩具狐梗","树ing沃克浣熊猎犬",
"维兹拉犬","魏玛犬",
"威尔士梗","威尔士跳猎犬","西高地白梗","惠比特犬",
"刚毛指示格里芬犬",
"约克夏梗"
]


patterns = [
    "白色",
    "黄色",
    "米色",
    "金色",
    "黑色",
    "棕色",
    "灰色",
    "纯色被毛",
    "双色被毛",
    "三色被毛",
    "黑白花",
    "棕白相间",
    "金白相间",
    "灰白相间",
    "大面积拼色",
    "不规则拼色",
    "斑点花纹",
    "马鞍状花纹",
    "背部马鞍色",
    "渐层毛色",
    "毛尖渐变色",
    "局部杂色",
    "混合毛色",
]



def generate_json_if_needed():
    if os.path.exists(JSON_PATH):
        print("Reusing existing dogs.json")
        with open(JSON_PATH,"r",encoding="utf-8") as f:
            return json.load(f)

    print("Generating dogs.json ...")

    data = []
    for i in range(TOTAL_COUNT):
        breed = random.choice(breeds)
        desc = f"{random.choice(patterns)}的{breed}"
        data.append({
            "id": i+1,
            "description": desc
        })

    os.makedirs(BASE_DIR, exist_ok=True)

    with open(JSON_PATH,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

    return data


########################################
# ========= Model loading =========
########################################

########################################
# ========= Prompt template =========
########################################

# Chinese prompts are used with the Qwen image-editing model.
PROMPT_TEMPLATE = """
严格UV贴图编辑模式。

这是一张3D模型的UV纹理贴图。中上为脸部贴图，形变是因为uv拉伸，不要改动结构。脸部左右上方分别为耳朵(贴近脸部的为正面)。脸部右边为口内以及下巴的贴图。脸部正下方是身体和尾巴，身体两侧为脚底贴图。

具体任务：
1. 将所有毛发颜色改为 <{dog_desc}> 风格。
2. 毛发纹理呈现自然真实的毛流方向与光影层次。
3. 五官的大小形状不可以改变。
4. 口内颜色形状不改变，只改变下巴边缘过渡色（应该与脖子一致）。
5. 保持鼻子形状大小，非常小的鼻子。
6. 保留脚底的爪垫。

要求：
- 真实自然的毛发质感
- 脸部，鼻子，眼睛, 嘴巴的布局，大小，边缘完全不可改变。
- 保持原始贴图分辨率与细节
- 只做毛发颜色与纹理替换
- 输出必须与原UV布局像素级对齐
"""

NEG_PROMPT = "shape change, different object, new design, different lighting, different background"



def parse_args(argv=None):
    p = argparse.ArgumentParser()

    p.add_argument("--start", type=int, default=1, help="start id (inclusive, 1-based)")
    p.add_argument("--end", type=int, default=None, help="end id (inclusive, 1-based). default = count")
    p.add_argument("--base_dir")
    p.add_argument("--input_texture_path")
    p.add_argument("--model_id")
    p.add_argument("--device")
    return parse_configured_args(p, "dataset_generation", argv)

########################################
# ========= Generation =========
########################################

def main():
    global BASE_DIR, JSON_PATH, TEXTURE_DIR, INPUT_TEXTURE_PATH
    args = parse_args()
    BASE_DIR = args.base_dir
    JSON_PATH = os.path.join(BASE_DIR, "dogs.json")
    TEXTURE_DIR = os.path.join(BASE_DIR, "texture")
    INPUT_TEXTURE_PATH = args.input_texture_path
    require_file(INPUT_TEXTURE_PATH, "base UV texture for optional data generation")
    import torch
    from PIL import Image
    from diffusers import QwenImageEditPlusPipeline
    print("Loading pipeline...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipeline.to(args.device)
    pipeline.set_progress_bar_config(disable=None)
    image1 = Image.open(INPUT_TEXTURE_PATH).convert("RGB")
    data = generate_json_if_needed()
    max_id = len(data)
    start_id = max(1, args.start)
    end_id = args.end if args.end is not None else max_id
    end_id = min(end_id, max_id)
    os.makedirs(TEXTURE_DIR, exist_ok=True)



    for idx in range(start_id, end_id + 1):
        item = data[idx - 1]  # IDs are one-based; list indices are zero-based.
        idx = item["id"]
        desc = item["description"]

        save_path = os.path.join(TEXTURE_DIR, f"{idx:06d}.png")

        if os.path.exists(save_path):
            print(f"Skip {idx} (exists)")
            continue

        prompt = PROMPT_TEMPLATE.format(dog_desc=desc)
        print(prompt)
        print(f"Generating {idx}...")

        inputs = {
            "image": [image1],
            "prompt": prompt,
            "generator": torch.manual_seed(idx),
            "true_cfg_scale": 3,
            "negative_prompt": NEG_PROMPT,
            "num_inference_steps": 20,
            "guidance_scale": 0.1,
            "num_images_per_prompt": 1,
        }

        with torch.inference_mode():
            output = pipeline(**inputs)

        output.images[0].save(save_path)
        print(f"Saved -> {save_path}")


if __name__ == "__main__":
    main()
