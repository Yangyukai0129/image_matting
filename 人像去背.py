import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils                    # mediapipe 繪圖功能
mp_selfie_segmentation = mp.solutions.selfie_segmentation  # mediapipe 自拍分割方法

name = 'monalisa'

# 載入圖片
img = cv2.imread(f'./image/{name}.jpg')  # 請將 'images2.jpg' 換成你的圖片檔名

# mediapipe 啟用自拍分割
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:

    # 確保圖片正確載入
    if img is None:
        print("Cannot open image")
        exit()

    # 調整圖片大小，保持長寬比
    height, width = img.shape[:2]
    aspect_ratio = width / height
    new_width = 520
    new_height = int(new_width / aspect_ratio)
    img_resized = cv2.resize(img, (new_width, new_height))

    # 將圖片轉換為RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 使用 selfie_segmentation 進行人像分割
    results = selfie_segmentation.process(img_rgb)  # 取得自拍分割結果
    
    # 設定區域條件 (背景與前景區域)
    condition = results.segmentation_mask > 0.9  # 判斷前景區域 (0.9 會更精確但可以試試不同的閾值)

    # 使用 Canny 邊緣檢測來提取邊緣
    edges = cv2.Canny(img_resized, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # 將邊緣圖像轉換為三通道圖像，以便後續處理

    # 使用高斯模糊平滑圖像，這樣背景會變得更加模糊
    background = cv2.GaussianBlur(img_resized, (15, 15), 0)

    # 合併邊緣圖像和背景圖像（這樣有助於突出人物邊緣）
    img_with_edges = cv2.bitwise_and(img_resized, edges)
    img_with_edges = cv2.bitwise_or(img_with_edges, background)

    # 建立透明背景的影像
    img_no_bg = np.zeros_like(img_resized, dtype=np.uint8)
    
    # 只保留主體區域
    img_no_bg[condition] = img_resized[condition]

    # 將圖像轉為 BGRA 格式，增加 alpha 通道處理透明度
    img_no_bg = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2BGRA)

    # 這裡增加你提到的灰度值根據條件設置透明度
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)  # 將圖像轉換為灰度圖
    for x in range(img_resized.shape[1]):  # w = width
        for y in range(img_resized.shape[0]):  # h = height
            if gray[y, x] > 200:
                img_no_bg[y, x, 3] = 255 - gray[y, x]  # 設置透明度 (alpha 通道)

    # 儲存處理後的圖片，帶有透明背景
    cv2.imwrite(f'./image/{name}_no_bg.png', img_no_bg)

    # # 顯示去背後的圖片
    # cv2.imshow('Processed Image', img_no_bg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()