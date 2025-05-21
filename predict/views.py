from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model
import numpy as np
import io
import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Modelni yuklash
MODEL_PATH = os.path.join(BASE_DIR, 'skin_cancer_model.h5')
model = load_model(MODEL_PATH)

# Klasslar nomlari va tavsiflari
class_names = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']
class_descriptions = [
    "Quyosh nurlari ta'sirida yillar davomida paydo bo‘ladigan dog‘.",
    "Bazal hujayrali karsinoma, teri saratoni turi.",
    "Saratonsiz teri o‘sishi.",
    "Fibroz teri shishi.",
    "Xol yoki melanotsitlarning xavfsiz ko‘payishi.",
    "Qon tomirlari hosil bo‘lgan xavfsiz o‘sishlar.",
    "Melanotsitlardan kelib chiqadigan jiddiy teri saratoni."
]

@csrf_exempt
def predict_view(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Faqat POST so‘rovlari qabul qilinadi!"}, status=400)

    if 'image' not in request.FILES:
        return JsonResponse({"error": "Rasm fayli topilmadi"}, status=400)

    try:
        # Rasm faylini o‘qish
        img_file = request.FILES['image'].read()
        img_bytes = io.BytesIO(img_file)

        # Rasmni PIL bilan o‘qish va o‘lchamini o‘zgartirish
        img = Image.open(img_bytes).convert('RGB')  # Ba'zida grayscale bo'lishi mumkin
        img = img.resize((224, 224))
        img_array = np.array(img)

        if img_array.ndim == 2:  # Grayscale holat
            img_array = np.stack((img_array,) * 3, axis=-1)

        img_array = np.expand_dims(img_array, axis=0) / 255.0
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        result = {
            "class": class_names[predicted_class],
            "description": class_descriptions[predicted_class]
        }
        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({"error": f"Server xatosi: {str(e)}"}, status=500)
