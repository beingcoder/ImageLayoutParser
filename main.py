import base64

import cv2
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Union
import uvicorn
import layoutparser as lp
import image2text as i2t

app = FastAPI()


class Image(BaseModel):
    name: str
    base64_content: Union[str, None] = None


model = lp.Detectron2LayoutModel(
    config_path="models/config2.yaml",
    model_path="models/model_final2.pth",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
    label_map={1: "TextRegion", 2: "ImageRegion", 3: "TableRegion", 4: "MathsRegion", 5: "SeparatorRegion",
               6: "OtherRegion"}
)

ocr_agent = lp.TesseractAgent(languages='eng')


@app.get("/")
async def root():
    return {"message": "a11y demo"}


@app.post("/a11ydemo")
async def a11y_demo(image_req: Image):
    image_byte = base64.b64decode(image_req.base64_content)
    image = cv2.imdecode(np.frombuffer(image_byte, dtype=np.uint8), cv2.IMREAD_COLOR)
    # image = image[..., ::-1]
    detection_recs = []
    layout = model.detect(image)

    for spot_rec in layout:
        if spot_rec.type == 'TextRegion':
            segment_img = spot_rec.crop_image(image)
            ocr_text = ocr_agent.detect(segment_img).strip()
            spot_rec.set(text=ocr_text + "\nAuto generated.", inplace=True)
        elif spot_rec.type == 'ImageRegion':
            segment_img = spot_rec.crop_image(image)
            image_cap = i2t.predict_step(segment_img)
            image_cap = image_cap[0]
            spot_rec.set(text=image_cap + "\nAuto generated.", inplace=True)
        else:
            spot_rec.set(text="", inplace=True)

        detection_recs.append({
            "coordinates": spot_rec.coordinates,
            "type": spot_rec.type,
            "text": str(spot_rec.text)
        })

    image_decorated = lp.draw_box(image, layout, show_element_type=True)
    image_decorated.save("decorated_" + image_req.name)

    return {
        "name": image_req.name,
        "segments": detection_recs,
        "llm_summary": "TODO"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        port=5009,
        host="0.0.0.0",
        reload=True
    )
