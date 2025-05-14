from datasets import Dataset, Features, Value, ClassLabel, Sequence, Image
import json
import PIL.Image as pil_image
from io import BytesIO
from tqdm import tqdm

json_paths = [
    
]

short_names = [
    # "mavis_math_metagen",
    # "mavis_math_rule_geo",
    # "k12_printing",
    # "iiit5k",
    # "hme100k",
    # "ai2d(gpt4v)",
    # "infographic_vqa",
    # "infographic(gpt4v)",
    # "lrv_chart",
    # "lrv_normal(filtered)",
    # "scienceqa(nona_context)",
    # "allava_instruct_vflan4v",
    # "allava_instruct_laion4v",
    # "textocr(gpt4v)",
    # "ai2d(internvl)",
    # "textcaps",
    # "ureader_qa", # need to re-upload
    # "ureader_cap", # need to re-upload
    # "ureader_ie", # need to re-upload
    # "ureader_kg", # need to re-upload
    # "vision_flan(filtered)",
    # "mathqa",
    # "geo3k",
    # "geo170k(qa)",
    # "geo170k(align)",
    # "sharegpt4v(coco)",
    # "sharegpt4v(knowledge)",
    # "sharegpt4v(llava)",
    # "sharegpt4v(sam)",
    # "CLEVR-Math(MathV360K)",
    # "FigureQA(MathV360K)",
    # "Geometry3K(MathV360K)",
    # "GeoQA+(MathV360K)",
    # "GEOS(MathV360K)",
    # "IconQA(MathV360K)",
    # "MapQA(MathV360K)",
    # "PMC-VQA(MathV360K)",
    # "Super-CLEVR(MathV360K)",
    # "TabMWP(MathV360K)",
    # "UniGeo(MathV360K)",
    # "VizWiz(MathV360K)",
    # "magpie_pro(qwen2_72b_st)",
    # "magpie_pro(l3_80b_st)",
    # "magpie_pro(l3_80b_mt)",
    # "image_textualization(filtered)",
    # "cambrian(filtered_gpt4vo)", # need to re-upload
    "sharegpt4o",
    "ai2d(cauldron,llava_format)",
    "aokvqa(cauldron,llava_format)",
    "chart2text(cauldron)",
    "chartqa(cauldron,llava_format)",
    "clevr(cauldron,llava_format)",
    "diagram_image_to_text(cauldron)",
    "dvqa(cauldron,llava_format)",
    "figureqa(cauldron,llava_format)",
    "geomverse(cauldron)",
    "hateful_memes(cauldron,llava_format)",
    "hitab(cauldron,llava_format)",
    "iam(cauldron)",
    "raven(cauldron)",
    "iconqa(cauldron,llava_format)",
    "infographic_vqa_llava_format",
    "intergps(cauldron,llava_format)",
    "mapqa(cauldron,llava_format)",
    "multihiertt(cauldron)",
    "rendered_text(cauldron)",
    "robut_sqa(cauldron)",
    "robut_wikisql(cauldron)",
    "robut_wtq(cauldron,llava_format)",
    "screen2words(cauldron)",
    "scienceqa(cauldron,llava_format)",
    "tabmwp(cauldron)",
    "tallyqa(cauldron,llava_format)",
    "st_vqa(cauldron,llava_format)",
    "tqa(cauldron,llava_format)",
    "visual7w(cauldron,llava_format)",
    "visualmrc(cauldron)",
    "vqarad(cauldron,llava_format)",
    "vsr(cauldron,llava_format)",
    "vistext(cauldron)",
    "websight(cauldron)"
]

def upload_data(json_path, short_name):
    def gen():
        if json_path.endswith(".jsonl"):
            with open(json_path, "r") as f:
                data = [json.loads(line) for line in f]
        else:
            with open(json_path, "r") as f:
                data = json.load(f)

        preview_index = 5
        idx = 0
        for item in tqdm(data):
            if preview_index > 0:
                preview_index -= 1
                print(item)
                continue

            try:
                if "image" in item:
                    image_path = f"/dataset/data/llava_data/{item['image']}"
                    try:
                        with open(image_path, "rb") as img_file:
                            image = pil_image.open(BytesIO(img_file.read()))
                    except:
                        print(f"Failed to load image {item['image']}")
                        continue
                else:
                    image = None

                item_id = item["id"] if "id" in item else f"{idx:06d}"
                yield {"id": item_id, "image": image, "conversations": item["conversations"], "data_source": short_name}
                idx += 1
                
            except Exception as e:
                print(e)
                continue


    hf_dataset = Dataset.from_generator(generator=gen, num_proc=32)
    hf_dataset.push_to_hub("lmms-lab/LLaVA-OneVision-Data", config_name=short_name, split="train")

for json_path, short_name in zip(json_paths, short_names):
    upload_data(json_path, short_name)