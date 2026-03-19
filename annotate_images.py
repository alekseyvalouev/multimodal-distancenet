# go through each folder with images from bww8 (with an upper limit on number of folders (say 50))
# annotate the images with landmarks using gemini, save in folder_name_meta.json
# the json should be formatted as follows: {"scenes" : [{0 : ["l1", "l2", "l3"], 1 : ["l4", "l5", "l6"], ...}]}

import os
import glob
from pathlib import Path
import json

from dotenv import load_dotenv  
from google import genai        

from pydantic import BaseModel
from typing import List

from tqdm import tqdm

from prompts import annotate_landmarks


MAX_FOLDERS = 50

class Landmarks(BaseModel):
    landmarks: List[str]

def setup_gemini():
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found in .env")
        return

    # Client picks up GEMINI_API_KEY from env, but we pass it explicitly too
    client = genai.Client(api_key=api_key)

    return client

def get_annotations(client, image_bytes):
    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[
            genai.types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            annotate_landmarks
        ],
        config={
            "response_mime_type": "application/json",
            "response_schema": Landmarks,
        },
    )
    return Landmarks.model_validate_json(response.text)

def main():
    client = setup_gemini()

    data_path = "/hdd/sacson"
    out_path = "/home/alekseyvalouev/goalnav/language-annotations"
    traj_paths = glob.glob(os.path.join(data_path, "*bww8*"), recursive=True)[1:MAX_FOLDERS+1]
    print(f"Found {len(traj_paths)} folders.")

    for i, traj_path in enumerate(traj_paths):
        if not os.path.isdir(traj_path):
            continue

        out_dict = {"landmarks" : {}}
        for image_path in tqdm(glob.glob(os.path.join(traj_path, "*.jpg")), desc=f"Annotating {traj_path.split('/')[-1]}"):
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            landmarks = get_annotations(client, image_bytes)
            out_dict["landmarks"][image_path.split("/")[-1]] = landmarks.landmarks

        with open(os.path.join(out_path, f"{traj_path.split('/')[-1]}_landmarks.json"), "w+") as f:
            json.dump(out_dict, f)

        print(f"Annotated {len(out_dict['landmarks'])} images for {traj_path.split('/')[-1]}")

if __name__ == "__main__":
    main()