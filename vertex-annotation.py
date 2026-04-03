import os
import glob
import json
from pathlib import Path
from tqdm import tqdm

from google.cloud import storage
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerationConfig

# User Settings
PROJECT_ID = "cat-gen-lang-proj"
LOCATION = "global"
BUCKET_NAME = "landmark-annotations"
MODEL_ID = "gemini-3.1-pro-preview"
LOCAL_DATA_PATH = "/hdd/sacson"
GCS_STAGING_PATH = "landmark_batch_job" # Folder inside your bucket

# Landmark Schema for Structured Output
LANDMARK_SCHEMA = {
    "type": "object",
    "properties": {
        "landmarks": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["landmarks"]
}

def upload_to_gcs(local_path, gcs_path):
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    return f"gs://{BUCKET_NAME}/{gcs_path}"

def create_batch_input():
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    scenes_train = [
        "Dec-06-2022-bww8_00000030_10", "Feb-09-2023-bww8-intloss_00000022_1", 
        "Jan-17-2023-bww8_00000001_0", "Dec-06-2022-bww8_00000037_0", "Feb-09-2023-bww8-intloss_00000031_4", 
        "Jan-17-2023-bww8_00000001_5", "Dec-07-2022-bww8_00000000_12", "Feb-09-2023-bww8-intloss_00000042_1", 
        "Jan-17-2023-bww8_00000002_10", "Dec-12-2022-bww8_00000036_0", "Feb-13-2023-bww8-intloss_00000009_3", 
        "Nov-17-2022-bww8_00000009_2", "Feb-09-2023-bww8-intloss_00000000_0", "Jan-12-2023-bww8_00000008_2", 
        "Nov-17-2022-bww8_00000012_0", "Jan-12-2023-bww8_00000009_29"
    ]
    scenes_test = [
        "Dec-06-2022-bww8_00000007_0", "Feb-09-2023-bww8-intloss_00000042_9", "Jan-12-2023-bww8_00000007_22",
        "Feb-03-2023-bww8-intloss_00000013_1", "Feb-14-2023-bww8-intloss_00000008_25"
    ]
    scenes_other = [
        "Feb-15-2023-cory1_00000000_0", "Feb-15-2023-cory1_00000006_4", "Feb-16-2023-cory1-intloss_00000021_1",
        "Feb-15-2023-cory1_00000004_6", "Feb-15-2023-cory1_00000006_5", "Feb-16-2023-cory1-intloss_00000023_0"
    ]
    scenes = scenes_train + scenes_test + scenes_other

    from prompts import annotate_landmarks # Your local prompt string
    
    jsonl_entries = []
    
    for scene in scenes:
        scene_dir = os.path.join(LOCAL_DATA_PATH, scene)
        if not os.path.isdir(scene_dir): continue
        
        image_files = glob.glob(os.path.join(scene_dir, "*.jpg"))
        
        for img_path in tqdm(image_files, desc=f"Uploading {scene}"):
            filename = os.path.basename(img_path)
            # 1. Upload image to GCS
            gcs_uri = upload_to_gcs(img_path, f"{GCS_STAGING_PATH}/images/{scene}/{filename}")
            
            # 2. Build JSONL request for this image
            # The structure follows the Gemini Batch API requirements
            request_item = {
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": annotate_landmarks},
                                {"file_data": {"mime_type": "image/jpeg", "file_uri": gcs_uri}}
                            ]
                        }
                    ],
                    "generation_config": {
                        "temperature": 0,
                        "response_mime_type": "application/json",
                        "response_schema": LANDMARK_SCHEMA
                    }
                }
            }
            jsonl_entries.append(request_item)

    # Save and upload the manifest JSONL
    manifest_local = "batch_input.jsonl"
    with open(manifest_local, "w") as f:
        for entry in jsonl_entries:
            f.write(json.dumps(entry) + "\n")
            
    manifest_gcs = upload_to_gcs(manifest_local, f"{GCS_STAGING_PATH}/input/manifest.jsonl")
    return manifest_gcs

def main():
    #print("Step 1: Preparing GCS data and Manifest...")
    #input_uri = create_batch_input()
    
    print(f"Step 2: Submitting Batch Job for {MODEL_ID}...")
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    
    # Using the Batch Prediction Job service
    batch_job = aiplatform.BatchPredictionJob.create(
        job_display_name="gemini_landmark_mass_annotation",
        model_name=f"publishers/google/models/{MODEL_ID}",
        instances_format="jsonl",
        predictions_format="jsonl",
        gcs_source=[f"gs://{BUCKET_NAME}/{GCS_STAGING_PATH}/input/manifest.jsonl"],
        gcs_destination_prefix=f"gs://{BUCKET_NAME}/{GCS_STAGING_PATH}/output/",
    )

    print(f"Job submitted! Job ID: {batch_job.name}")
    print(f"View progress in Console: https://console.cloud.google.com/vertex-ai/batch-predictions?project={PROJECT_ID}")

if __name__ == "__main__":
    main()