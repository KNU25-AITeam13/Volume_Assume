# -*- coding: utf-8 -*-
r"""
.\.venv\Scripts\Activate.ps1

[Simple Scale Update]
- Fallback 모드(수저 없음) 시: 깊이맵은 원본 그대로 두고,
  최종 계산된 '부피(Volume)' 값에만 0.1을 곱하여 출력.

실행 예시:
  python volume_test.py --mode yolo --image scene.jpg --depth depth.npy
"""

import argparse
import sys
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    _YOLO_AVAILABLE = False


# ---------------------------------------------------------
# [설정] 수저류 실제 크기 (단위: cm)
# ---------------------------------------------------------
DEFAULT_SIZES = {
    'spoon': 18.0,       
    'fork': 19.0,        
    'knife': 22.0,       
    'chopsticks': 21.0,  
}

# ---------------------------------------------------------
# [유틸] 객체 크기 측정
# ---------------------------------------------------------
def get_max_pixel_dimension(mask: np.ndarray) -> int:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return 0
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    return int(max(rect[1]))

# ---------------------------------------------------------
# [유틸] 화각 기반 초점거리
# ---------------------------------------------------------
def get_focal_length_from_fov(image_width: int, fov_deg: float = 72.0) -> float:
    fov_rad = np.deg2rad(fov_deg)
    return (image_width / 2.0) / np.tan(fov_rad / 2.0)

# ---------------------------------------------------------
# [핵심] 부피 계산
# ---------------------------------------------------------
def volume_calculation_core(
        Z_scene: np.ndarray,
        food_mask: np.ndarray,
        bg_mask_candidate: np.ndarray,
        ref_px_len: float,
        ref_real_cm: float,
        density_g_per_ml: float = 1.0,
        is_fallback_mode: bool = False
    ):
    
    H, W = Z_scene.shape
    
    # 1. 바닥면(Ground) 찾기
    bg_mask = bg_mask_candidate & (~food_mask) & (~np.isnan(Z_scene))
    if bg_mask.sum() < 100:
        kernel = np.ones((15,15), np.uint8)
        dilated_food = cv2.dilate(food_mask.astype(np.uint8), kernel, iterations=2)
        bg_mask = (dilated_food > 0) & (~food_mask) & (~np.isnan(Z_scene))
    if not bg_mask.any(): bg_mask = (~np.isnan(Z_scene))

    # 2. 평면 피팅
    grid_y, grid_x = np.indices((H, W))
    p_ys, p_xs, p_zs = grid_y[bg_mask], grid_x[bg_mask], Z_scene[bg_mask]

    if p_xs.size > 50:
        A_mat = np.column_stack((p_xs, p_ys, np.ones_like(p_xs)))
        plane_coeffs, _, _, _ = np.linalg.lstsq(A_mat, p_zs, rcond=None)
    else:
        median_depth = np.median(p_zs) if p_zs.size > 0 else 1.0
        plane_coeffs = np.array([0.0, 0.0, median_depth])

    # 3. 스케일 결정 (깊이맵 자체는 건드리지 않음)
    dist_ref_m = float(np.median(p_zs)) if p_zs.size > 0 else 1.0

    if not is_fallback_mode and ref_px_len > 0 and ref_real_cm > 0:
        f_px = (ref_px_len * dist_ref_m) / (ref_real_cm / 100.0)
        method_str = f"기준물체 사용 (수저 {ref_real_cm}cm)"
    else:
        f_px = get_focal_length_from_fov(W, fov_deg=72.0)
        ref_px_len = 0
        method_str = "기준물체 없음 (화각 72도)"

    # 4. 부피 적분 (원본 Depth 사용)
    valid_food = food_mask & (~np.isnan(Z_scene))
    f_ys, f_xs = np.where(valid_food)
    z_measured = Z_scene[valid_food]
    
    z_base = np.column_stack((f_xs, f_ys, np.ones_like(f_xs))) @ plane_coeffs
    heights_m = np.maximum(0.0, z_base - z_measured)
    
    max_height_cm = float(np.max(heights_m) * 100.0) if heights_m.size > 0 else 0.0
    
    volumes_m3 = heights_m * ((z_measured / f_px) ** 2)
    total_volume_ml = np.sum(volumes_m3) * 1e6
    
    # [수정됨] 사용자의 요청: Fallback 모드일 때 최종 부피에만 0.1 곱하기
    if is_fallback_mode:
        print(" [Scale] 기준물체 부재로 최종 부피에 0.1을 곱합니다.")
        total_volume_ml = total_volume_ml * 0.1
        method_str
    
    return {
        "volume_ml": total_volume_ml,
        "mass_g": total_volume_ml * density_g_per_ml,
        "max_height_cm": max_height_cm,
        "method": method_str,
        "avg_depth_m": dist_ref_m
    }

# ---------------------------------------------------------
# [YOLO]
# ---------------------------------------------------------
CUTLERY_LIKE = {'spoon', 'fork', 'knife', 'chopsticks'} 
PLATE_LIKE = {'plate', 'bowl', 'cup', 'wine glass', 'tray'} 
FOOD_LIKE  = {
    'food', 'rice', 'noodles', 'pizza', 'sandwich', 'salad', 'cake', 'donut',
    'banana','apple','orange','broccoli','carrot','hot dog','burger','steak','bread'
}

def yolo_inference(image_path, yolo_weights, depth_shape_hw):
    if not _YOLO_AVAILABLE: raise RuntimeError("ultralytics 미설치")
    H, W = depth_shape_hw
    model = YOLO(yolo_weights)
    results = model(image_path, imgsz=640, conf=0.15, verbose=False)
    
    if not results or results[0].masks is None:
        return {}, np.zeros((H,W),bool), [], []

    res = results[0]
    masks = res.masks.data.cpu().numpy()
    clses = res.boxes.cls.cpu().numpy().astype(int)
    names = res.names

    detected_refs = {}      
    max_area_per_cls = {}   
    food_mask_accum = np.zeros((H, W), dtype=bool)
    bg_candidate_mask = np.zeros((H, W), dtype=bool)
    picked_log = [] 
    food_names = [] 

    for mi, ci in zip(masks, clses):
        cls_name = names[ci]
        m_resized = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        current_area = m_resized.sum()

        if cls_name in FOOD_LIKE:
            food_mask_accum |= m_resized
            picked_log.append(('Food', cls_name))
            food_names.append(cls_name) 
            
        elif cls_name in CUTLERY_LIKE:
            bg_candidate_mask |= m_resized
            if cls_name not in max_area_per_cls or current_area > max_area_per_cls[cls_name]:
                detected_refs[cls_name] = m_resized
                max_area_per_cls[cls_name] = current_area
            picked_log.append(('Ref', cls_name))

        elif cls_name in PLATE_LIKE or cls_name == 'dining table':
            bg_candidate_mask |= m_resized
            picked_log.append(('BG', cls_name))

    return detected_refs, food_mask_accum, bg_candidate_mask, picked_log, list(set(food_names))

# ---------------------------------------------------------
# [Main]
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synth", "yolo"], default="synth")
    parser.add_argument("--image", type=str)
    parser.add_argument("--depth", type=str)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--yolo_weights", type=str, default="yolov8x-seg.pt")
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--spoon-cm", type=float, default=None)
    parser.add_argument("--fork-cm", type=float, default=None)

    args = parser.parse_args()

    if args.mode == "synth":
        print("합성 모드 미지원. --mode yolo 사용.")
        sys.exit(0)

    if not args.image or not args.depth:
        print("Error: --image, --depth 필수")
        sys.exit(1)
        
    if args.depth.lower().endswith('.npy'):
        Z_scene = np.load(args.depth).astype(np.float32)
    else:
        Z_scene = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED).astype(np.float32)
    Z_scene *= args.depth_scale

    detected_refs, food_mask, bg_candidate_mask, logs, food_names = yolo_inference(args.image, args.yolo_weights, Z_scene.shape)

    ref_px_len = 0.0
    ref_real_cm = 0.0
    is_fallback = False
    
    sizes = DEFAULT_SIZES.copy()
    if args.spoon_cm: sizes['spoon'] = args.spoon_cm
    if args.fork_cm: sizes['fork'] = args.fork_cm

    priority_order = ['spoon', 'fork', 'knife', 'chopsticks']
    found_ref = None
    
    for item in priority_order:
        if item in detected_refs:
            px_len = get_max_pixel_dimension(detected_refs[item])
            if px_len > 10:
                found_ref = item
                ref_px_len = px_len
                ref_real_cm = sizes.get(item, 20.0)
                break
    
    if found_ref:
        print(f" 기준 물체 감지됨: [{found_ref}]")
    else:
        print(" 기준 물체 없음 -> DepthPro 화각 모드")
        is_fallback = True

    res = volume_calculation_core(
        Z_scene=Z_scene,
        food_mask=food_mask,
        bg_mask_candidate=bg_candidate_mask,
        ref_px_len=ref_px_len,
        ref_real_cm=ref_real_cm,
        density_g_per_ml=args.density,
        is_fallback_mode=is_fallback
    )

    food_display = ", ".join(food_names) if food_names else "감지되지 않음 (Unidentified)"

    print("\n" + "="*40)
    print(f" [분석 결과]")
    print(f"  - 음식 종류: {food_display}")
    print(f"  - 추정 부피 : {res['volume_ml']:.1f} ml")
    print(f"  - 추정 질량 : {res['mass_g']:.1f} g")
    print(f"  - 최대 높이 : {res['max_height_cm']:.2f} cm")
    print(f"  - 계산 방식 : {res['method']}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()
