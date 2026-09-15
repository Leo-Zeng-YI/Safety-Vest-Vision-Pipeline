import os
import cv2
import torch
import numpy as np
import time
import logging
from datetime import datetime
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def setup_logger():
    # 生成带时间戳的日志文件名
    log_filename = f"sam3_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 配置日志，使其同时输出到文件和控制台
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        # force=True，强制覆盖 sam3 或 torch 预设的日志体系
        force=True, 
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_filename

def main():
    # ==========================================
    # 0. 初始化日志与计时器
    # ==========================================
    log_filename = setup_logger()
    global_start_time = time.time()
    
    # 初始化统计数据
    stats = {
        "total_images_processed": 0,
        "total_persons_detected": 0,
        "total_vests_cropped": 0
    }

    # ==========================================
    # 1. 基础与路径配置
    # ==========================================
    input_base = "../Data"
    output_base = "./Data_Vest_Crops" 
    splits = ["train", "val", "test"]
    model_path = "sam3.pt"
    
    prompt_texts = ["safety vest", "reflecting clothing"] 
    
    HEAD_CUTOFF_RATIO = 0.20
    MIN_VEST_AREA = 50
    SCALE_RATIO = 1.05 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"[*] 启动反光衣 SAM3 分割裁剪引擎，使用设备: {device}")

    for split in splits:
        os.makedirs(os.path.join(output_base, split), exist_ok=True)

    # ==========================================
    # 2. 加载 SAM 3 模型
    # ==========================================
    logging.info("[*] 正在加载 SAM 3 模型...")
    model = build_sam3_image_model(checkpoint_path=model_path)
    model.to(device=device)  
    processor = Sam3Processor(model)
    logging.info("[*] 模型加载完成！")

    # ==========================================
    # 3. 遍历处理数据集
    # ==========================================
    for split in splits:
        img_dir_in = os.path.join(input_base, "images", split)
        lbl_dir_in = os.path.join(input_base, "labels", split)
        save_dir_out = os.path.join(output_base, split)
        
        if not os.path.exists(img_dir_in) or not os.path.exists(lbl_dir_in):
            continue
            
        logging.info("====================================")
        logging.info(f"[*] 正在处理: {split.upper()}")
        logging.info("====================================")
        
        for img_name in os.listdir(img_dir_in):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): 
                continue
                
            img_path_in = os.path.join(img_dir_in, img_name)
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            lbl_path_in = os.path.join(lbl_dir_in, lbl_name)
            
            if not os.path.exists(lbl_path_in):
                continue
                
            image_raw = cv2.imread(img_path_in)
            if image_raw is None: continue
            
            stats["total_images_processed"] += 1
            h_img, w_img = image_raw.shape[:2]
            image_pil_full = Image.fromarray(cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB))
            
            with open(lbl_path_in, 'r') as f:
                lines = f.readlines()
            
            person_idx = 0
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                class_id = int(parts[0])
                if class_id == 0:
                    stats["total_persons_detected"] += 1
                    
                    cx, cy = float(parts[1]) * w_img, float(parts[2]) * h_img
                    w, h = float(parts[3]) * w_img, float(parts[4]) * h_img
                    
                    w_enlarged = w * SCALE_RATIO
                    h_enlarged = h * SCALE_RATIO
                    
                    crop_x1 = max(0, int(cx - w_enlarged / 2))
                    crop_y1 = max(0, int(cy - h_enlarged / 2))
                    crop_x2 = min(w_img, int(cx + w_enlarged / 2))
                    crop_y2 = min(h_img, int(cy + h_enlarged / 2))
                    
                    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                        continue
                        
                    crop_pil = image_pil_full.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    
                    if device == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            state = processor.set_image(crop_pil)
                    else:
                        state = processor.set_image(crop_pil)
                    
                    best_vest_mask_global = None
                    
                    for prompt in prompt_texts:
                        if device == "cuda":
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                res = processor.set_text_prompt(state=state, prompt=prompt)
                        else:
                            res = processor.set_text_prompt(state=state, prompt=prompt)
                            
                        if "masks" in res:
                            masks = res["masks"]
                            if isinstance(masks, torch.Tensor):
                                masks = masks.cpu().float().numpy()
                                
                            combined_local_mask = np.any(masks > 0.0, axis=0).squeeze().astype(np.uint8) * 255
                            
                            if np.sum(combined_local_mask == 255) > MIN_VEST_AREA:
                                global_mask = np.zeros((h_img, w_img), dtype=np.uint8)
                                mask_h, mask_w = combined_local_mask.shape
                                global_mask[crop_y1:crop_y1+mask_h, crop_x1:crop_x1+mask_w] = combined_local_mask
                                best_vest_mask_global = global_mask
                                break 
                    
                    if best_vest_mask_global is not None:
                        original_y1 = max(0, int(cy - h/2))
                        head_limit = int(original_y1 + h * HEAD_CUTOFF_RATIO)
                        best_vest_mask_global[0:head_limit, :] = 0
                        
                        contours, _ = cv2.findContours(best_vest_mask_global, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours: continue
                        largest_cnt = max(contours, key=cv2.contourArea)
                        
                        mask_area = cv2.contourArea(largest_cnt)
                        box_area = w_enlarged * h_enlarged
                        
                        M = cv2.moments(largest_cnt)
                        if M["m00"] != 0:
                            c_y = int(M["m01"] / M["m00"])
                        else:
                            c_y = 0
                        
                        if MIN_VEST_AREA <= mask_area <= (box_area * 0.75) and c_y <= (crop_y1 + h_enlarged * 0.75):
                            
                            final_mask = np.zeros_like(best_vest_mask_global)
                            cv2.drawContours(final_mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)
                            
                            vx, vy, vw, vh = cv2.boundingRect(largest_cnt)
                            
                            aspect_ratio = vh / float(vw) if vw > 0 else 0
                            vest_bottom_y = vy + vh 
                            max_bottom_allowed = crop_y1 + (h_enlarged * 0.65)
                            
                            if aspect_ratio < 1.8 and vest_bottom_y < max_bottom_allowed and vw >= 30 and vh >= 30: 
                                
                                vest_pixels_only = cv2.bitwise_and(image_raw, image_raw, mask=final_mask)
                                final_vest_crop = vest_pixels_only[vy:vy+vh, vx:vx+vw]
                                
                                save_name = f"{os.path.splitext(img_name)[0]}_p{person_idx}_vest.jpg"
                                cv2.imwrite(os.path.join(save_dir_out, save_name), final_vest_crop)
                                
                                stats["total_vests_cropped"] += 1
                    person_idx += 1

    # ==========================================
    # 4. 计算指标与日志输出
    # ==========================================
    total_time_seconds = time.time() - global_start_time
    total_images = stats["total_images_processed"]
    avg_time_per_image = (total_time_seconds / total_images) if total_images > 0 else 0

    logging.info("\n====================================")
    logging.info("[*] 步骤一完成！处理结果摘要：")
    logging.info("====================================")
    logging.info(f"  - 总处理图片数: {total_images} 张")
    logging.info(f"  - 检测到人员 (Person) 目标数: {stats['total_persons_detected']} 个")
    logging.info(f"  - 成功提取有效反光衣裁剪图: {stats['total_vests_cropped']} 张")
    logging.info(f"  - 总处理耗时: {total_time_seconds:.2f} 秒 ({total_time_seconds / 60:.2f} 分钟)")
    logging.info(f"  - 平均单张图片处理耗时: {avg_time_per_image:.3f} 秒")
    logging.info(f"[*] 裁剪数据已保存在 {output_base}")
    logging.info(f"[*] 完整运行日志已保存至: {log_filename}")

if __name__ == "__main__":
    main()
    