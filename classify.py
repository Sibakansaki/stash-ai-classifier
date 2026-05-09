import base64
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

IMAGE_PATH = "test_covers/001.jpg"
GROUPS_PATH = "groups.json"


def encode_image(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def main():
    with open(GROUPS_PATH, "r", encoding="utf-8") as f:
        groups = json.load(f)

    image_base64 = encode_image(IMAGE_PATH)

    prompt = f"""
你是一個影片收藏分類助手。

任務：
根據封面圖片，替影片選擇「唯一一個主群組」。

規則：
1. 一部影片只能選一個群組。
2. 優先從現有 groups.json 裡的 groups 選擇。
3. 如果真的沒有適合的群組，才可以建議新群組。
4. 不要輸出色情細節描述，只判斷分類。
5. 必須輸出純 JSON，不要加解釋文字。

groups.json：
{json.dumps(groups, ensure_ascii=False, indent=2)}

輸出格式：
{{
  "group": "群組名稱",
  "confidence": 0,
  "reason": "簡短理由",
  "is_new_group": false,
  "suggested_new_group": null
}}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_base64}",
                    },
                ],
            }
        ],
    )

    print(response.output_text)


if __name__ == "__main__":
    main()
